"""FuzzPuppy Clip Merge 1:1 faithful local Python/OpenCV/ONNX engine.

Implements the complete FuzzPuppy pipeline (`eI` seam analysis & dual-gate hard-cut
protection, `eS` multi-timestamp dense 8x8 grid pyramid LK + 64-iteration edge-aware
diffusion + B-clip 3-keyframe subject tracking, `eT` composed non-rigid alignment with
0.8s quintic release & anti-ghosting overlap blend, `V` rife422lite ONNX / local RIFE
symmetric bridge, and `eP` 30ms smoothstep audio crossfade).
"""

from __future__ import annotations

import importlib
import inspect
import math
import threading
from pathlib import Path
from typing import Any, Optional

import cv2 as cv
import numpy as np
import torch


_ONNX_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}
_ONNX_LOCK = threading.Lock()
_MODEL_DIR = Path(__file__).resolve().parent / "models"


def _clamp(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else (hi if val > hi else val)


def _smoothstep(x: float) -> float:
    t = _clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _quintic_release(elapsed: float, duration: float = 0.8) -> float:
    if duration <= 1e-6:
        return 0.0
    a = _clamp(elapsed / duration, 0.0, 1.0)
    return 1.0 - a * a * a * (10.0 + a * (-15.0 + 6.0 * a))


def _ensure_u8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim >= 3 and frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(np.round(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8))


def _resize_u8(img_u8: np.ndarray, width: int, height: int) -> np.ndarray:
    if img_u8.shape[1] == width and img_u8.shape[0] == height:
        return img_u8
    interp = cv.INTER_AREA if (width < img_u8.shape[1] or height < img_u8.shape[0]) else cv.INTER_LINEAR
    return cv.resize(img_u8, (width, height), interpolation=interp)


def _fit_frame_to_size(img_u8: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox/pillarbox fit matching FuzzPuppy's canvas `fit: 'contain'`."""
    h, w = img_u8.shape[:2]
    if w == target_w and h == target_h:
        return img_u8
    scale = min(target_w / max(1, w), target_h / max(1, h))
    nw = max(1, min(target_w, int(round(w * scale))))
    nh = max(1, min(target_h, int(round(h * scale))))
    resized = _resize_u8(img_u8, nw, nh)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


# ---------------------------------------------------------------------------
# Stage 1 (eI): Normalized Luma, Structure Error, 4-DOF Similarity, Coverage
# ---------------------------------------------------------------------------

def normalize_luma_el(img_u8: np.ndarray) -> np.ndarray:
    """Exact port of FuzzPuppy `el()`: Rec.709 luma, zero-mean, std >= 0.08."""
    f = img_u8.astype(np.float32)
    luma = (0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]) / 255.0
    mean = float(luma.mean())
    std = max(0.08, float(np.sqrt(((luma - mean) ** 2).mean())))
    return ((luma - mean) / std).astype(np.float32)


def structure_error_ec(
    luma_a: np.ndarray,
    luma_b: np.ndarray,
    transform: dict[str, float],
    step: int = 3,
) -> float:
    """Exact port of FuzzPuppy `ec()`: clipped L1 structural error under inverse similarity."""
    h, w = luma_a.shape
    if h <= 12 or w <= 12:
        return float(np.minimum(1.0, np.abs(luma_a - luma_b)).mean())
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ca = math.cos(transform["angle"]) / transform["scale"]
    sa = math.sin(transform["angle"]) / transform["scale"]
    ys = np.arange(6, h - 6, step, dtype=np.float32)
    xs = np.arange(6, w - 6, step, dtype=np.float32)
    if ys.size == 0 or xs.size == 0:
        return float(np.minimum(1.0, np.abs(luma_a - luma_b)).mean())
    gx, gy = np.meshgrid(xs, ys)
    rx = gx - cx - transform["x"]
    ry = gy - cy - transform["y"]
    ux = ca * rx + sa * ry + cx
    uy = -sa * rx + ca * ry + cy
    oob = (ux < 2.0) | (uy < 2.0) | (ux >= w - 2.0) | (uy >= h - 2.0)
    ux_c = np.clip(ux, 0.0, w - 1.001)
    uy_c = np.clip(uy, 0.0, h - 1.001)
    x0 = np.floor(ux_c).astype(np.int32)
    y0 = np.floor(uy_c).astype(np.int32)
    fx = ux_c - x0
    fy = uy_c - y0
    val = (
        (luma_b[y0, x0] * (1.0 - fx) + luma_b[y0, x0 + 1] * fx) * (1.0 - fy)
        + (luma_b[y0 + 1, x0] * (1.0 - fx) + luma_b[y0 + 1, x0 + 1] * fx) * fy
    )
    ref = luma_a[gy.astype(np.int32), gx.astype(np.int32)]
    diff = np.minimum(1.0, np.abs(ref - val))
    diff[oob] = 1.0
    return float(diff.mean())


def fit_similarity_ed(luma_a: np.ndarray, luma_b: np.ndarray) -> tuple[dict[str, float], float]:
    """Exact port of FuzzPuppy `ed()`: coarse grid + coordinate descent similarity fit."""
    h, w = luma_a.shape
    best = {"x": 0.0, "y": 0.0, "scale": 1.0, "angle": 0.0}
    best_err = structure_error_ec(luma_a, luma_b, best, step=3)
    for oy in range(-8, 9, 4):
        for ox in range(-12, 13, 4):
            cand = {"x": float(ox), "y": float(oy), "scale": 1.0, "angle": 0.0}
            err = structure_error_ec(luma_a, luma_b, cand, step=4)
            if err < best_err:
                best = cand
                best_err = err
    best_err = structure_error_ec(luma_a, luma_b, best, step=3)
    max_x, max_y = 0.12 * w, 0.12 * h
    for o in (2.0, 1.0, 0.5, 0.25):
        for _ in range(5):
            improved = False
            for key in ("x", "y", "scale", "angle"):
                step_val = 0.01 * o if key == "scale" else (0.008 * o if key == "angle" else o)
                for sign in (-1.0, 1.0):
                    cand = dict(best)
                    cand[key] = best[key] + sign * step_val
                    if (
                        abs(cand["x"]) > max_x
                        or abs(cand["y"]) > max_y
                        or cand["scale"] < 0.92
                        or cand["scale"] > 1.08
                        or abs(cand["angle"]) > 0.065
                    ):
                        continue
                    err = structure_error_ec(luma_a, luma_b, cand, step=3)
                    if err + 1e-5 < best_err:
                        best = cand
                        best_err = err
                        improved = True
            if not improved:
                break
    return best, best_err


def compute_rgb_stats_ef(img_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact port of FuzzPuppy `ef()`: non-black (R+G+B >= 12) RGB mean and std."""
    flat = img_u8.reshape(-1, 3)[::4].astype(np.float32)
    mask = flat.sum(axis=1) >= 12.0
    valid = flat[mask]
    if len(valid) == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    mean = valid.mean(axis=0)
    var = np.maximum(0.0, (valid ** 2).mean(axis=0) - mean ** 2)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def compute_color_params(img_a_u8: np.ndarray, img_b_warped_u8: np.ndarray) -> dict[str, np.ndarray]:
    """Exact FuzzPuppy color matching: gain in [0.8, 1.25], offset in [-30, 30]."""
    mean_a, std_a = compute_rgb_stats_ef(img_a_u8)
    mean_b, std_b = compute_rgb_stats_ef(img_b_warped_u8)
    gain = np.clip(std_a / np.maximum(8.0, std_b), 0.8, 1.25).astype(np.float32)
    offset = np.clip(mean_a - mean_b * gain, -30.0, 30.0).astype(np.float32)
    return {"gain": gain, "offset": offset}


def warp_similarity_eh(
    img_u8: np.ndarray,
    transform: dict[str, float],
    amount: float = 1.0,
    color: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Exact port of FuzzPuppy `eh()`: global similarity warp + optional bounded RGB correction."""
    h, w = img_u8.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    sc = 1.0 + (transform["scale"] - 1.0) * amount
    ang = transform["angle"] * amount
    ca = math.cos(ang) / sc
    sa = math.sin(ang) / sc
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rx = gx - cx - transform["x"] * amount
    ry = gy - cy - transform["y"] * amount
    ux = np.clip(ca * rx + sa * ry + cx, 0.0, w - 1.001).astype(np.float32)
    uy = np.clip(-sa * rx + ca * ry + cy, 0.0, h - 1.001).astype(np.float32)
    warped = cv.remap(img_u8, ux, uy, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if color is None or amount <= 1e-6:
        return warped
    wf = warped.astype(np.float32)
    gain = color["gain"].reshape(1, 1, 3)
    offset = color["offset"].reshape(1, 1, 3)
    adj = wf + amount * (wf * (gain - 1.0) + offset)
    out = np.where(wf > 3.0, adj, wf)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _block_match_grid_eu(luma_src: np.ndarray, luma_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact port of FuzzPuppy `eu()` over all 8x8 grid centers."""
    h, w = luma_src.shape
    gw = int(math.ceil(w / 8.0))
    gh = int(math.ceil(h / 8.0))
    dx_map = np.zeros((gh, gw), dtype=np.int32)
    dy_map = np.zeros((gh, gw), dtype=np.int32)
    conf_map = np.zeros((gh, gw), dtype=np.float32)

    ys = np.arange(gh) * 8 + 4
    xs = np.arange(gw) * 8 + 4
    valid_y = np.where((ys >= 9) & (ys < h - 9))[0]
    valid_x = np.where((xs >= 9) & (xs < w - 9))[0]
    if valid_y.size == 0 or valid_x.size == 0:
        return dx_map, dy_map, conf_map

    cy = ys[valid_y]
    cx = xs[valid_x]
    # Precompute horizontal gradient sum over [-3..3]x[-3..3]
    grad_x = np.abs(luma_src[:, 1:] - luma_src[:, :-1])
    s_grad = np.zeros((len(cy), len(cx)), dtype=np.float32)
    for oy in range(-3, 4):
        for ox in range(-3, 4):
            s_grad += grad_x[np.ix_(cy + oy, cx + ox)]

    # Extract 4x4 strided-2 reference patches [ny, nx, 4, 4]
    py_offs = np.array([-3, -1, 1, 3])
    px_offs = np.array([-3, -1, 1, 3])
    src_patch = np.empty((len(cy), len(cx), 16), dtype=np.float32)
    k = 0
    for oy in py_offs:
        for ox in px_offs:
            src_patch[:, :, k] = luma_src[np.ix_(cy + oy, cx + ox)]
            k += 1

    best_err = np.full((len(cy), len(cx)), np.inf, dtype=np.float32)
    sec_err = np.full((len(cy), len(cx)), np.inf, dtype=np.float32)
    best_dx = np.zeros((len(cy), len(cx)), dtype=np.int32)
    best_dy = np.zeros((len(cy), len(cx)), dtype=np.int32)

    for sy in range(-5, 6):
        for sx in range(-5, 6):
            d = np.zeros((len(cy), len(cx)), dtype=np.float32)
            k = 0
            for oy in py_offs:
                y_idx = cy + oy + sy
                for ox in px_offs:
                    x_idx = cx + ox + sx
                    d += np.abs(src_patch[:, :, k] - luma_dst[np.ix_(y_idx, x_idx)])
                    k += 1
            d /= 16.0
            better = d < best_err
            sec_update = (~better) & (d < sec_err)
            sec_err = np.where(better, best_err, np.where(sec_update, d, sec_err))
            best_err = np.where(better, d, best_err)
            best_dx = np.where(better, sx, best_dx)
            best_dy = np.where(better, sy, best_dy)

    conf = (
        np.clip((sec_err - best_err) / np.maximum(0.02, sec_err) * 3.0, 0.0, 1.0)
        * np.clip(s_grad / 8.0, 0.0, 1.0)
        * np.clip(1.0 - best_err / 0.55, 0.0, 1.0)
    )
    dx_map[np.ix_(valid_y, valid_x)] = best_dx
    dy_map[np.ix_(valid_y, valid_x)] = best_dy
    conf_map[np.ix_(valid_y, valid_x)] = conf
    return dx_map, dy_map, conf_map


def _block_match_single_eu(luma_src: np.ndarray, luma_dst: np.ndarray, cx: int, cy: int) -> tuple[int, int]:
    h, w = luma_src.shape
    if cx < 9 or cy < 9 or cx >= w - 9 or cy >= h - 9:
        return 99, 99
    patch = luma_src[cy - 3:cy + 4:2, cx - 3:cx + 4:2]
    best_err = float("inf")
    bi, bl = 0, 0
    for sy in range(-5, 6):
        y0 = cy + sy - 3
        for sx in range(-5, 6):
            x0 = cx + sx - 3
            err = float(np.abs(patch - luma_dst[y0:y0 + 7:2, x0:x0 + 7:2]).mean())
            if err < best_err:
                best_err = err
                bi, bl = sx, sy
    return bi, bl


def block_match_coverage_eu(luma_a: np.ndarray, luma_b_warped: np.ndarray) -> float:
    """Exact port of FuzzPuppy `eI` bidirectional 8x8 block-match coverage."""
    h, w = luma_a.shape
    dx1, dy1, conf1 = _block_match_grid_eu(luma_a, luma_b_warped)
    gh, gw = dx1.shape
    good = 0
    for gy in range(gh):
        cy = gy * 8 + 4
        if cy < 9 or cy >= h - 9:
            continue
        for gx in range(gw):
            cx = gx * 8 + 4
            if cx < 9 or cx >= w - 9:
                continue
            if conf1[gy, gx] <= 0.15:
                continue
            dx = int(dx1[gy, gx])
            dy = int(dy1[gy, gx])
            if abs(dx) == 5 or abs(dy) == 5:
                continue
            bx, by = _block_match_single_eu(luma_b_warped, luma_a, cx + dx, cy + dy)
            if math.hypot(dx + bx, dy + by) <= 1.5:
                good += 1
    return float(good) / max(1, gw * gh)


def _sample_frame_at_time(frames_u8: np.ndarray, fps: float, t_sec: float) -> tuple[int, np.ndarray]:
    idx = int(_clamp(round(t_sec * fps), 0, len(frames_u8) - 1))
    return idx, frames_u8[idx]


def analyze_seam_eI(
    frames_a_u8: np.ndarray,
    frames_b_u8: np.ndarray,
    fps: float,
    max_search_sec: float = 0.5,
    overlap_frames: int = 4,
    structure_threshold: float = 0.38,
    coverage_threshold: float = 0.08,
    preserve_duration: bool = False,
) -> dict[str, Any]:
    """Exact port of FuzzPuppy `eI`: 81-pair seam search, similarity fit, and dual safety gate."""
    fps = float(_clamp(float(fps), 12.0, 60.0))
    n_a, n_b = len(frames_a_u8), len(frames_b_u8)
    h, w = frames_a_u8.shape[1:3]
    dur_a = n_a / fps
    dur_b = n_b / fps

    probe_w = 128
    probe_h = max(24, int(round(h / max(1, w) * probe_w)))

    def get_probe(frames: np.ndarray, t_sec: float) -> tuple[int, np.ndarray, np.ndarray]:
        idx, fr = _sample_frame_at_time(frames, fps, t_sec)
        im = _resize_u8(_fit_frame_to_size(fr, w, h), probe_w, probe_h)
        return idx, im, normalize_luma_el(im)

    if preserve_duration or max_search_sec <= 0.0 or overlap_frames <= 0:
        s_overlap = 0.0
        idx_a, im_a, luma_a = get_probe(frames_a_u8, (n_a - 1) / fps)
        idx_b, im_b, luma_b = get_probe(frames_b_u8, 0.0)
        tr, err = fit_similarity_ed(luma_a, luma_b)
        u_trim_a, u_trim_b = 0.0, 0.0
        u_time_a, u_time_b = (n_a - 1) / fps, 0.0
    else:
        s_overlap = min(float(overlap_frames) / fps, dur_a / 4.0, dur_b / 4.0)
        r_a = max(0.0, min(max_search_sec, dur_a - s_overlap - 1.0 / fps))
        r_b = max(0.0, min(max_search_sec, dur_b - s_overlap - 1.0 / fps))
        t_as = [dur_a - s_overlap / 2.0 - r_a + i * r_a / 8.0 for i in range(9)]
        t_bs = [s_overlap / 2.0 + i * r_b / 8.0 for i in range(9)]
        fa_list = [(t, *get_probe(frames_a_u8, t)) for t in t_as]
        fb_list = [(t, *get_probe(frames_b_u8, t)) for t in t_bs]

        identity = {"x": 0.0, "y": 0.0, "scale": 1.0, "angle": 0.0}
        coarse_pairs = []
        for ta, ia, ima, la in fa_list:
            trim_a = max(0.0, dur_a - ta - s_overlap / 2.0)
            for tb, ib, imb, lb in fb_list:
                trim_b = max(0.0, tb - s_overlap / 2.0)
                c_score = structure_error_ec(la, lb, identity, step=4) + 0.08 * (trim_a + trim_b)
                coarse_pairs.append((c_score, trim_a, trim_b, ta, tb, ia, ib, ima, imb, la, lb))
        coarse_pairs.sort(key=lambda x: x[0])

        zero_candidates = [p for p in coarse_pairs if abs(p[1]) < 0.001 and abs(p[2]) < 0.001]
        cands = list(coarse_pairs[:10])
        if zero_candidates:
            z = zero_candidates[0]
            if not any(abs(c[1] - z[1]) < 1e-6 and abs(c[2] - z[2]) < 1e-6 for c in cands):
                cands.append(z)

        refined = []
        for _, trim_a, trim_b, ta, tb, ia, ib, ima, imb, la, lb in cands:
            tr_c, err_c = fit_similarity_ed(la, lb)
            refined.append((err_c + 0.08 * (trim_a + trim_b), err_c, trim_a, trim_b, ta, tb, ia, ib, ima, imb, la, lb, tr_c))
        refined.sort(key=lambda x: x[0])
        _, err, u_trim_a, u_trim_b, u_time_a, u_time_b, idx_a, idx_b, im_a, im_b, luma_a, luma_b, tr = refined[0]

    warped_b = warp_similarity_eh(im_b, tr, amount=1.0)
    cov = block_match_coverage_eu(luma_a, normalize_luma_el(warped_b))

    mean_a, _ = compute_rgb_stats_ef(im_a)
    mean_b, _ = compute_rgb_stats_ef(warped_b)
    color_diff = float(np.abs(mean_a - mean_b).mean() / 255.0)

    # Estimate pre-seam vs post-seam camera motion velocity difference
    vel = []
    for is_b, (frames_src, t_cur, luma_cur) in enumerate(((frames_a_u8, u_time_a, luma_a), (frames_b_u8, u_time_b, luma_b))):
        t_nbr = _clamp(t_cur + (1.0 / fps if is_b else -1.0 / fps), 0.0, (len(frames_src) - 1) / fps)
        _, _, luma_nbr = get_probe(frames_src, t_nbr)
        tr_v, _ = fit_similarity_ed(luma_cur, luma_nbr)
        sign = -1.0 if is_b else 1.0
        vel.append((tr_v["x"] * sign, tr_v["y"] * sign))
    motion_diff = float(math.hypot(vel[0][0] - vel[1][0], vel[0][1] - vel[1][1]))

    safe = bool((err < structure_threshold) and (cov > coverage_threshold))
    raw_score = round(
        100.0
        * _clamp(
            0.3 * (1.0 - _clamp(math.hypot(tr["x"], tr["y"]) / 16.0 + 4.0 * abs(tr["scale"] - 1.0) + 4.0 * abs(tr["angle"]), 0.0, 1.0))
            + 0.3 * (1.0 - _clamp(err / 0.6, 0.0, 1.0))
            + 0.2 * (1.0 - _clamp(color_diff / 0.25, 0.0, 1.0))
            + 0.2 * (1.0 - _clamp(motion_diff / 5.0, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    score = raw_score if safe else min(39, raw_score)
    quality = ("Excellent" if score >= 85 else ("Good" if score >= 65 else "Fair")) if safe else "Poor"
    color_params = compute_color_params(im_a, warped_b)

    trim_a_eff = max(0.0, u_trim_a) if safe else 0.0
    trim_b_eff = max(0.0, u_trim_b) if safe else 0.0
    overlap_eff = s_overlap if safe else 0.0

    full_transform = {
        "x": tr["x"] * (w / float(probe_w)),
        "y": tr["y"] * (h / float(probe_h)),
        "scale": tr["scale"],
        "angle": tr["angle"],
    }
    return {
        "width": w,
        "height": h,
        "fps": fps,
        "dur_a": dur_a,
        "dur_b": dur_b,
        "trimA": trim_a_eff,
        "trimB": trim_b_eff,
        "overlap": overlap_eff,
        "seam": dur_a - trim_a_eff - overlap_eff / 2.0,
        "duration": dur_a - trim_a_eff + dur_b - trim_b_eff - overlap_eff,
        "transform": full_transform,
        "color": color_params,
        "structureError": round(err, 5),
        "flowCoverage": round(cov, 5),
        "motionDifference": round(motion_diff, 4),
        "colorDifference": round(color_diff, 5),
        "score": int(score),
        "quality": quality,
        "safe": safe,
        "best_a_idx": int(idx_a),
        "best_b_idx": int(idx_b),
        "preserve_duration": bool(preserve_duration),
    }


# ---------------------------------------------------------------------------
# Stage 2 (eS): Dense 8x8 Grid Pyramid LK + Edge-Aware Diffusion + B-Tracking
# ---------------------------------------------------------------------------

def dense_grid_lk_ey(luma_a: np.ndarray, luma_b: np.ndarray, step: int = 8) -> dict[str, Any]:
    """Vectorized faithful port of FuzzPuppy `ey()` + `ep()`: 8x8 grid 4-level pyramid bidirectional LK."""
    h, w = luma_a.shape
    gw = int(math.ceil(w / float(step)))
    gh = int(math.ceil(h / float(step)))
    dx_grid = np.zeros((gh, gw), dtype=np.float32)
    dy_grid = np.zeros((gh, gw), dtype=np.float32)
    conf_grid = np.zeros((gh, gw), dtype=np.float32)

    # Normalize luma to uint8 for OpenCV multi-level pyramidal LK (preserves local contrast identically)
    a_u8 = np.clip((luma_a + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip((luma_b + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)

    xs = (np.arange(gw, dtype=np.float32) * step + step * 0.5)
    ys = (np.arange(gh, dtype=np.float32) * step + step * 0.5)
    gx, gy = np.meshgrid(xs, ys)
    valid_border = (gx >= 5) & (gy >= 5) & (gx < w - 5) & (gy < h - 5)
    pts = np.stack([gx[valid_border], gy[valid_border]], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    if len(pts) == 0:
        return {"width": gw, "height": gh, "step": step, "dx": dx_grid, "dy": dy_grid, "confidence": conf_grid, "coverage": 0.0}

    max_level = 0
    mw, mh = w, h
    while max_level < 3 and min(mw, mh) >= 48:
        mw //= 2
        mh //= 2
        max_level += 1

    lk_params = dict(
        winSize=(9, 9),
        maxLevel=max_level,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 12, 0.02),
        minEigThreshold=8e-5,
    )
    fwd, st_f, _ = cv.calcOpticalFlowPyrLK(a_u8, b_u8, pts, None, **lk_params)
    bwd, st_b, _ = cv.calcOpticalFlowPyrLK(b_u8, a_u8, fwd, None, **lk_params)

    # Precompute structure-tensor min eigenvalue (exact ep() texture metric) on luma_a
    gx_im = cv.Sobel(luma_a, cv.CV_32F, 1, 0, ksize=1) * 0.5
    gy_im = cv.Sobel(luma_a, cv.CV_32F, 0, 1, ksize=1) * 0.5
    sxx = cv.boxFilter(gx_im * gx_im, cv.CV_32F, (9, 9), normalize=False)
    sxy = cv.boxFilter(gx_im * gy_im, cv.CV_32F, (9, 9), normalize=False)
    syy = cv.boxFilter(gy_im * gy_im, cv.CV_32F, (9, 9), normalize=False)
    min_eig = (sxx + syy - np.hypot(sxx - syy, 2.0 * sxy)) / 162.0

    max_disp = min(40.0, 0.12 * w)
    pts2 = pts.reshape(-1, 2)
    fwd2 = fwd.reshape(-1, 2)
    bwd2 = bwd.reshape(-1, 2)
    disp = fwd2 - pts2
    disp_norm = np.hypot(disp[:, 0], disp[:, 1])
    fb_err = np.hypot(bwd2[:, 0] - pts2[:, 0], bwd2[:, 1] - pts2[:, 1])

    ok = (
        (st_f.reshape(-1) == 1)
        & (st_b.reshape(-1) == 1)
        & np.isfinite(disp_norm)
        & (disp_norm <= max_disp)
        & (fb_err <= 1.2)
        & (fwd2[:, 0] >= 5)
        & (fwd2[:, 1] >= 5)
        & (fwd2[:, 0] < w - 5)
        & (fwd2[:, 1] < h - 5)
    )

    valid_indices = np.flatnonzero(valid_border)
    good_k = np.flatnonzero(ok)

    # Vectorized 9x9 patch L1 error check on passing points
    if good_k.size > 0:
        px = pts2[good_k, 0]
        py = pts2[good_k, 1]
        fx = fwd2[good_k, 0]
        fy = fwd2[good_k, 1]
        tex = min_eig[py.astype(np.int32), px.astype(np.int32)]

        # Evaluate 9x9 bilinear patch error between luma_a(px+dx, py+dy) and luma_b(fx+dx, fy+dy)
        offs = np.arange(-4, 5, dtype=np.float32)
        ox_g, oy_g = np.meshgrid(offs, offs)
        ox_f = ox_g.reshape(1, 81)
        oy_f = oy_g.reshape(1, 81)
        pa = luma_a[
            (py[:, None] + oy_f).astype(np.int32),
            (px[:, None] + ox_f).astype(np.int32),
        ]
        bx_q = np.clip(fx[:, None] + ox_f, 0.0, w - 1.001)
        by_q = np.clip(fy[:, None] + oy_f, 0.0, h - 1.001)
        bx0 = np.floor(bx_q).astype(np.int32)
        by0 = np.floor(by_q).astype(np.int32)
        bfx = bx_q - bx0
        bfy = by_q - by0
        pb = (
            (luma_b[by0, bx0] * (1.0 - bfx) + luma_b[by0, bx0 + 1] * bfx) * (1.0 - bfy)
            + (luma_b[by0 + 1, bx0] * (1.0 - bfx) + luma_b[by0 + 1, bx0 + 1] * bfx) * bfy
        )
        patch_err = np.abs(pa - pb).mean(axis=1)

        keep = (tex >= 8e-5) & (patch_err <= 0.45)
        kept = good_k[keep]
        if kept.size > 0:
            conf_vals = (
                np.clip(tex[keep] / 0.002, 0.0, 1.0)
                * np.clip(1.0 - patch_err[keep] / 0.45, 0.0, 1.0)
                * np.clip(1.0 - fb_err[kept] / 1.2, 0.0, 1.0)
            )
            flat_idx = valid_indices[kept]
            dx_grid.reshape(-1)[flat_idx] = disp[kept, 0]
            dy_grid.reshape(-1)[flat_idx] = disp[kept, 1]
            conf_grid.reshape(-1)[flat_idx] = conf_vals.astype(np.float32)

    coverage = float((conf_grid > 0.15).mean())
    return {
        "width": gw,
        "height": gh,
        "step": step,
        "dx": dx_grid,
        "dy": dy_grid,
        "confidence": conf_grid,
        "coverage": coverage,
    }


def _motion_consensus(flows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact port of FuzzPuppy `eS` multi-timestamp median + inlier consensus."""
    if len(flows) == 1:
        return flows[0]
    gh, gw = flows[0]["dx"].shape
    dx_stack = np.stack([f["dx"] for f in flows], axis=0)
    dy_stack = np.stack([f["dy"] for f in flows], axis=0)
    cf_stack = np.stack([f["confidence"] for f in flows], axis=0)

    out_dx = np.zeros((gh, gw), dtype=np.float32)
    out_dy = np.zeros((gh, gw), dtype=np.float32)
    out_cf = np.zeros((gh, gw), dtype=np.float32)
    n_flows = float(len(flows))

    for y in range(gh):
        for x in range(gw):
            mask = cf_stack[:, y, x] > 0.12
            if not np.any(mask):
                continue
            dxs = dx_stack[mask, y, x]
            dys = dy_stack[mask, y, x]
            cfs = cf_stack[mask, y, x]
            med_x = np.sort(dxs)[len(dxs) // 2]
            med_y = np.sort(dys)[len(dys) // 2]
            inl = np.hypot(dxs - med_x, dys - med_y) < 1.5
            if not np.any(inl):
                continue
            w_sum = float(cfs[inl].sum())
            out_dx[y, x] = float((dxs[inl] * cfs[inl]).sum() / w_sum)
            out_dy[y, x] = float((dys[inl] * cfs[inl]).sum() / w_sum)
            out_cf[y, x] = w_sum / n_flows

    return {
        "width": gw,
        "height": gh,
        "step": flows[0]["step"],
        "dx": out_dx,
        "dy": out_dy,
        "confidence": out_cf,
        "coverage": float((out_cf > 0.15).mean()),
    }


def _sample_rgb_bilinear(img_f32: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img_f32.shape[:2]
    xc = np.clip(xs, 0.0, w - 1.001)
    yc = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(xc).astype(np.int32)
    y0 = np.floor(yc).astype(np.int32)
    fx = (xc - x0)[..., None]
    fy = (yc - y0)[..., None]
    return (
        (img_f32[y0, x0] * (1.0 - fx) + img_f32[y0, x0 + 1] * fx) * (1.0 - fy)
        + (img_f32[y0 + 1, x0] * (1.0 - fx) + img_f32[y0 + 1, x0 + 1] * fx) * fy
    )


def edge_aware_diffuse_ev(flow: dict[str, Any], guide_rgb_u8: np.ndarray, iterations: int = 64) -> dict[str, Any]:
    """Vectorized exact port of FuzzPuppy `ev()`: 4-neighbor edge-aware 64-iteration Jacobi propagation."""
    gw, gh, step = flow["width"], flow["height"], flow["step"]
    guide_f = guide_rgb_u8.astype(np.float32)

    gx, gy = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
    cx = (gx + 0.5) * step
    cy = (gy + 0.5) * step
    node_rgb = _sample_rgb_bilinear(guide_f, cx, cy)  # [gh, gw, 3]

    # Precompute 4 directional edge weights: left(-1,0), right(+1,0), up(0,-1), down(0,+1)
    weights = {}
    for name, (dc, dr) in (("L", (-1, 0)), ("R", (1, 0)), ("U", (0, -1)), ("D", (0, 1))):
        nx = gx + dc
        ny = gy + dr
        in_bounds = (nx >= 0) & (nx < gw) & (ny >= 0) & (ny < gh)
        nx_c = np.clip(nx, 0, gw - 1).astype(np.int32)
        ny_c = np.clip(ny, 0, gh - 1).astype(np.int32)
        h_sq = ((node_rgb - node_rgb[ny_c, nx_c]) ** 2).sum(axis=-1)
        u_max = np.zeros((gh, gw), dtype=np.float32)
        prev_sample = node_rgb
        for k in range(1, 5):
            cur_sample = _sample_rgb_bilinear(guide_f, (gx + 0.5 + dc * k / 4.0) * step, (gy + 0.5 + dr * k / 4.0) * step)
            step_diff = np.abs(cur_sample - prev_sample).max(axis=-1)
            u_max = np.maximum(u_max, step_diff)
            prev_sample = cur_sample
        m = np.exp(-h_sq / 2352.0 - (u_max ** 2) / 968.0).astype(np.float32)
        m = np.where(in_bounds & (m > 0.01), m, 0.0).astype(np.float32)
        weights[name] = m

    w_deg = np.maximum(1.0, weights["L"] + weights["R"] + weights["U"] + weights["D"])
    base_dx = flow["dx"].astype(np.float32)
    base_dy = flow["dy"].astype(np.float32)
    base_cf = flow["confidence"].astype(np.float32)
    anchored = base_cf > 0.15
    anchor_w = np.where(anchored, 4.0 * base_cf, 0.0).astype(np.float32)

    s_dx = base_dx.copy()
    s_dy = base_dy.copy()
    s_cf = base_cf.copy()

    for _ in range(iterations):
        num_x = base_dx * anchor_w
        num_y = base_dy * anchor_w
        den = anchor_w.copy()

        # Left neighbor (col - 1)
        if gw > 1:
            wl = weights["L"][:, 1:] * s_cf[:, :-1]
            num_x[:, 1:] += s_dx[:, :-1] * wl
            num_y[:, 1:] += s_dy[:, :-1] * wl
            den[:, 1:] += wl

            wr = weights["R"][:, :-1] * s_cf[:, 1:]
            num_x[:, :-1] += s_dx[:, 1:] * wr
            num_y[:, :-1] += s_dy[:, 1:] * wr
            den[:, :-1] += wr

        if gh > 1:
            wu = weights["U"][1:, :] * s_cf[:-1, :]
            num_x[1:, :] += s_dx[:-1, :] * wu
            num_y[1:, :] += s_dy[:-1, :] * wu
            den[1:, :] += wu

            wd = weights["D"][:-1, :] * s_cf[1:, :]
            num_x[:-1, :] += s_dx[1:, :] * wd
            num_y[:-1, :] += s_dy[1:, :] * wd
            den[:-1, :] += wd

        valid = den > 1e-6
        s_dx = np.where(valid, num_x / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        s_dy = np.where(valid, num_y / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        prop_cf = np.minimum(0.65, (den / w_deg) * 0.985)
        s_cf = np.where(valid, np.where(anchored, base_cf, prop_cf), 0.0).astype(np.float32)

    return {
        "width": gw,
        "height": gh,
        "step": step,
        "dx": s_dx,
        "dy": s_dy,
        "confidence": s_cf,
        "coverage": float((s_cf > 0.15).mean()),
    }


def _sample_flow_grid_em(flow: dict[str, Any], xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact port of FuzzPuppy `em()`: bilinear lookup on the 8x8 flow grid."""
    gw, gh, step = flow["width"], flow["height"], flow["step"]
    rx = np.clip(xs / float(step) - 0.5, 0.0, max(0.0, gw - 1.001)).astype(np.float32)
    ry = np.clip(ys / float(step) - 0.5, 0.0, max(0.0, gh - 1.001)).astype(np.float32)
    dx = cv.remap(flow["dx"], rx, ry, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    dy = cv.remap(flow["dy"], rx, ry, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    cf = cv.remap(flow["confidence"], rx, ry, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return dx, dy, cf


def _compose_flow(base_flow: dict[str, Any], track_flow: dict[str, Any]) -> dict[str, Any]:
    """Exact port of FuzzPuppy `eS` B-tracking flow composition `m(p + c(p))`."""
    gw, gh, step = base_flow["width"], base_flow["height"], base_flow["step"]
    gx, gy = np.meshgrid((np.arange(gw, dtype=np.float32) + 0.5) * step, (np.arange(gh, dtype=np.float32) + 0.5) * step)
    s_dx, s_dy, s_cf = _sample_flow_grid_em(track_flow, gx, gy)
    c_dx, c_dy, c_cf = _sample_flow_grid_em(base_flow, gx + s_dx, gy + s_dy)
    out_cf = (c_cf * np.clip(s_cf / 0.15, 0.0, 1.0)).astype(np.float32)
    return {
        "width": gw,
        "height": gh,
        "step": step,
        "dx": c_dx.astype(np.float32),
        "dy": c_dy.astype(np.float32),
        "confidence": out_cf,
        "coverage": float((out_cf > 0.15).mean()),
    }


def build_subject_alignment_eS(
    frames_a_u8: np.ndarray,
    frames_b_u8: np.ndarray,
    analysis: dict[str, Any],
    release_seconds: float = 0.8,
) -> dict[str, Any]:
    """Exact port of FuzzPuppy `eS`: 384px 3-timestamp LK consensus + edge diffusion + B-tracking."""
    w, h, fps = analysis["width"], analysis["height"], analysis["fps"]
    trim_a, trim_b, overlap = analysis["trimA"], analysis["trimB"], analysis["overlap"]
    dur_a, dur_b = analysis["dur_a"], analysis["dur_b"]
    color = analysis["color"]

    aw = min(384, w)
    ah = max(24, int(round(h * aw / float(w))))
    tr_s = {
        "x": analysis["transform"]["x"] * (aw / float(w)),
        "y": analysis["transform"]["y"] * (ah / float(h)),
        "scale": analysis["transform"]["scale"],
        "angle": analysis["transform"]["angle"],
    }

    def get_a(t_sec: float) -> np.ndarray:
        _, fr = _sample_frame_at_time(frames_a_u8, fps, t_sec)
        return _resize_u8(_fit_frame_to_size(fr, w, h), aw, ah)

    def get_b_aligned(t_sec: float) -> np.ndarray:
        _, fr = _sample_frame_at_time(frames_b_u8, fps, t_sec)
        im = _resize_u8(_fit_frame_to_size(fr, w, h), aw, ah)
        return warp_similarity_eh(im, tr_s, amount=1.0, color=color)

    if overlap > 1e-5:
        c_start = dur_a - trim_a - overlap
        pairs = []
        for frac in (1.0 / 6.0, 0.5, 5.0 / 6.0):
            n_t = overlap * frac
            pairs.append((get_a(c_start + n_t), get_b_aligned(trim_b + n_t)))
        flows = [dense_grid_lk_ey(normalize_luma_el(pa), normalize_luma_el(pb)) for pa, pb in pairs]
        mid_a, mid_b = pairs[1]
        consensus = _motion_consensus(flows)
    else:
        mid_a = get_a((len(frames_a_u8) - 1) / fps)
        mid_b = get_b_aligned(trim_b)
        consensus = dense_grid_lk_ey(normalize_luma_el(mid_a), normalize_luma_el(mid_b))

    diffused_m = edge_aware_diffuse_ev(consensus, mid_a, iterations=64)

    # Compute alignment residual on mid_b warped by diffused_m
    gx, gy = np.meshgrid(np.arange(aw, dtype=np.float32), np.arange(ah, dtype=np.float32))
    fdx, fdy, fcf = _sample_flow_grid_em(diffused_m, gx, gy)
    w_scale = np.clip(fcf / 0.15, 0.0, 1.0)
    mx = np.clip(gx + fdx * w_scale, 0.0, aw - 1.001).astype(np.float32)
    my = np.clip(gy + fdy * w_scale, 0.0, ah - 1.001).astype(np.float32)
    warped_mid_b = cv.remap(mid_b, mx, my, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    residual = float(np.abs(warped_mid_b.astype(np.float32) - mid_a.astype(np.float32)).mean())

    release_p = min(float(release_seconds), max(0.1, dur_b - trim_b - overlap))
    keys = [{"time": overlap, "flow": diffused_m}]
    luma_mid_b = normalize_luma_el(mid_b)
    for frac in (1.0 / 3.0, 2.0 / 3.0, 1.0):
        t_rel = overlap + release_p * frac
        b_t = get_b_aligned(min(trim_b + t_rel, max(0.0, dur_b - 0.001)))
        c_flow = edge_aware_diffuse_ev(dense_grid_lk_ey(normalize_luma_el(b_t), luma_mid_b), b_t, iterations=64)
        keys.append({"time": t_rel, "flow": _compose_flow(diffused_m, c_flow)})

    return {
        "width": aw,
        "height": ah,
        "keys": keys,
        "coverage": float(consensus["coverage"]),
        "residual": round(residual, 4),
        "release": float(release_p),
    }


def _interp_alignment_flow(subject_align: dict[str, Any], t_in_b: float) -> dict[str, Any]:
    keys = subject_align["keys"]
    if t_in_b <= keys[0]["time"]:
        return keys[0]["flow"]
    for idx in range(1, len(keys)):
        if keys[idx]["time"] >= t_in_b:
            k0, k1 = keys[idx - 1], keys[idx]
            frac = _smoothstep((t_in_b - k0["time"]) / max(1e-6, k1["time"] - k0["time"]))
            f0, f1 = k0["flow"], k1["flow"]
            return {
                "width": f0["width"],
                "height": f0["height"],
                "step": f0["step"],
                "dx": (f0["dx"] * (1.0 - frac) + f1["dx"] * frac).astype(np.float32),
                "dy": (f0["dy"] * (1.0 - frac) + f1["dy"] * frac).astype(np.float32),
                "confidence": (f0["confidence"] * (1.0 - frac) + f1["confidence"] * frac).astype(np.float32),
            }
    return keys[-1]["flow"]


def warp_composed_eT(
    frame_b_u8: np.ndarray,
    transform: dict[str, float],
    flow: dict[str, Any],
    align_width: int,
    amount: float,
    color: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Vectorized exact port of FuzzPuppy `eT` composed global similarity + local non-rigid flow warp."""
    if amount <= 1e-6:
        return frame_b_u8
    h, w = frame_b_u8.shape[:2]
    scale_ratio = w / float(align_width)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    sc = 1.0 + (transform["scale"] - 1.0) * amount
    ang = transform["angle"] * amount
    ca_o = math.cos(ang) / sc
    sa_o = math.sin(ang) / sc

    ca_f = math.cos(transform["angle"]) * transform["scale"]
    sa_f = math.sin(transform["angle"]) * transform["scale"]
    ca_inv = math.cos(transform["angle"]) / transform["scale"]
    sa_inv = math.sin(transform["angle"]) / transform["scale"]

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx = gx - cx - transform["x"] * amount
    fy = gy - cy - transform["y"] * amount
    vx = ca_o * fx + sa_o * fy
    vy = -sa_o * fx + ca_o * fy

    mx = (ca_f * vx - sa_f * vy + cx + transform["x"]) / scale_ratio
    my = (sa_f * vx + ca_f * vy + cy + transform["y"]) / scale_ratio
    fdx, fdy, fcf = _sample_flow_grid_em(flow, mx, my)
    w_loc = np.clip(fcf / 0.15, 0.0, 1.0) * amount * scale_ratio

    src_x = np.clip(vx + cx + (ca_inv * fdx + sa_inv * fdy) * w_loc, 0.0, w - 1.001).astype(np.float32)
    src_y = np.clip(vy + cy + (-sa_inv * fdx + ca_inv * fdy) * w_loc, 0.0, h - 1.001).astype(np.float32)
    warped = cv.remap(frame_b_u8, src_x, src_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if color is None:
        return warped
    wf = warped.astype(np.float32)
    gain = color["gain"].reshape(1, 1, 3)
    offset = color["offset"].reshape(1, 1, 3)
    adj = wf + amount * (wf * (gain - 1.0) + offset)
    out = np.where(wf > 3.0, adj, wf)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def blend_overlap_antighost(frame_a_u8: np.ndarray, frame_b_u8: np.ndarray, alpha: float) -> np.ndarray:
    """Exact port of FuzzPuppy `eT` difference-aware overlap cross-dissolve (avoids ghosting when max RGB diff >= 40)."""
    af = frame_a_u8.astype(np.float32)
    bf = frame_b_u8.astype(np.float32)
    max_diff = np.abs(af - bf).max(axis=-1, keepdims=True)
    steep_t = _clamp((alpha - 0.35) / 0.30, 0.0, 1.0)
    steep_w = steep_t * steep_t * (3.0 - 2.0 * steep_t)
    w = np.where(max_diff < 40.0, alpha, steep_w).astype(np.float32)
    return np.clip(np.round(af * (1.0 - w) + bf * w), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 3 (V): RIFE Inference (rife422lite ONNX primary + local RIFE fallback)
# ---------------------------------------------------------------------------

def default_onnx_rife_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    custom_nodes_dir = repo_root.parent
    comfy_root = custom_nodes_dir.parent
    candidates = [
        _MODEL_DIR / "rife422lite_fp32.onnx",
        comfy_root / "models" / "rife" / "rife422lite_fp32.onnx",
        custom_nodes_dir / "ComfyUI-Acheng-SeamBridge" / "models" / "rife422lite_fp32.onnx",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return _MODEL_DIR / "rife422lite_fp32.onnx"


def _get_onnx_session(model_path: Path):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or ["CPUExecutionProvider"]
    key = (str(model_path.resolve()), tuple(providers))
    with _ONNX_LOCK:
        sess = _ONNX_SESSION_CACHE.get(key)
        if sess is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
            _ONNX_SESSION_CACHE[key] = sess
    return sess


def run_rife_onnx(frame0_u8: np.ndarray, frame1_u8: np.ndarray, count: int, model_path: Optional[Path] = None) -> np.ndarray:
    """Exact port of FuzzPuppy `V()`: rife422lite ONNX direct endpoint inference with 32-multiple edge padding."""
    path = model_path or default_onnx_rife_path()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX RIFE model not found: {path}")
    sess = _get_onnx_session(path)
    h, w = frame0_u8.shape[:2]
    pw = ((w + 31) // 32) * 32
    ph = ((h + 31) // 32) * 32

    def prep(fr: np.ndarray) -> np.ndarray:
        x = fr.astype(np.float32) / 255.0
        if ph > h or pw > w:
            x = np.pad(x, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
        return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])

    img0 = prep(frame0_u8)
    img1 = prep(frame1_u8)
    flow_div = np.array([(pw - 1) / 2.0, (ph - 1) / 2.0], dtype=np.float32)
    gx = np.linspace(-1.0, 1.0, pw, dtype=np.float32) if pw > 1 else np.zeros(1, dtype=np.float32)
    gy = np.linspace(-1.0, 1.0, ph, dtype=np.float32) if ph > 1 else np.zeros(1, dtype=np.float32)
    grid = np.ascontiguousarray(np.stack(np.meshgrid(gx, gy), axis=0)[None].astype(np.float32))

    frames = []
    for k in range(count):
        t = np.array([[[[(k + 1.0) / (count + 1.0)]]]], dtype=np.float32)
        res = sess.run(
            ["frame"],
            {
                "img0": img0,
                "img1": img1,
                "timestep": t,
                "tenFlow_div": flow_div,
                "backwarp_tenGrid": grid,
            },
        )[0][0]
        if not np.isfinite(res).all():
            raise RuntimeError("ONNX RIFE produced non-finite frame values.")
        fr = np.transpose(res[:, :h, :w], (1, 2, 0)).clip(0.0, 1.0)
        frames.append(np.round(fr * 255.0).astype(np.uint8))
    return np.stack(frames, axis=0)


def run_rife_local_pth(frame0_u8: np.ndarray, frame1_u8: np.ndarray, count: int, checkpoint: str = "rife49.pth") -> np.ndarray:
    """Fallback to local ComfyUI-Frame-Interpolation RIFE VFI node."""
    from nodes import NODE_CLASS_MAPPINGS

    backend = NODE_CLASS_MAPPINGS.get("RIFE VFI")
    if backend is None:
        raise RuntimeError("ComfyUI-Frame-Interpolation RIFE VFI is not installed.")
    function = getattr(backend(), backend.FUNCTION)
    module = inspect.getmodule(backend)
    loader = getattr(module, "load_file_from_github_release", None)
    config = importlib.import_module(loader.__module__) if loader is not None else None
    root = getattr(config, "get_ckpt_container_path", None)
    versions = getattr(module, "CKPT_NAME_VER_DICT", {})
    if root is None or checkpoint not in versions or not (Path(root("rife")) / checkpoint).is_file():
        raise RuntimeError(f"Local RIFE checkpoint not found: {checkpoint}")
    endpoints = torch.from_numpy(np.stack([frame0_u8, frame1_u8], axis=0).astype(np.float32) / 255.0)
    options = {
        "ckpt_name": checkpoint,
        "frames": endpoints,
        "multiplier": count + 1,
        "fast_mode": True,
        "ensemble": False,
        "dtype": "float32",
        "torch_compile": False,
        "batch_size": 1,
        "clear_cache_after_n_frames": 10,
        "scale_factor": 1.0,
    }
    params = inspect.signature(function).parameters
    out = function(**{k: v for k, v in options.items() if k in params})[0]
    if out.shape != (count + 2, *endpoints.shape[1:]):
        raise RuntimeError("Local RIFE returned unexpected frame shape.")
    mid = out[1:-1].detach().cpu().clamp(0.0, 1.0).numpy()
    return np.round(mid * 255.0).astype(np.uint8)


def run_rife_bridge(
    frame0_u8: np.ndarray,
    frame1_u8: np.ndarray,
    count: int,
    backend_choice: str = "auto",
) -> tuple[np.ndarray, str]:
    """Run RIFE interpolation using FuzzPuppy's rife422lite ONNX (preferred) or local ComfyUI RIFE."""
    errors = []
    if backend_choice in ("auto", "rife422lite_onnx", "rife422lite_fp32.onnx"):
        try:
            frames = run_rife_onnx(frame0_u8, frame1_u8, count)
            return frames, "onnx:rife422lite_fp32"
        except Exception as err:
            errors.append(f"onnx:{err}")
            if backend_choice in ("rife422lite_onnx", "rife422lite_fp32.onnx"):
                raise
    pth_candidates = (
        [backend_choice]
        if backend_choice.endswith(".pth")
        else ["rife426.pth", "rife49.pth", "rife47.pth"]
    )
    for ckpt in pth_candidates:
        try:
            frames = run_rife_local_pth(frame0_u8, frame1_u8, count, checkpoint=ckpt)
            return frames, f"pth:{ckpt}"
        except Exception as err:
            errors.append(f"{ckpt}:{err}")
    raise RuntimeError("All RIFE backends failed: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Stage 4 (eT + eP): Full Pairwise Clip Merge (Frames + Audio)
# ---------------------------------------------------------------------------

def merge_audio_eP(
    audio_a: Optional[np.ndarray],
    audio_b: Optional[np.ndarray],
    sample_rate: int,
    v_sec: float,
    b_sec: float,
    m_sec: float,
    trim_b_sec: float,
    overlap_sec: float,
    total_sec: float,
) -> Optional[np.ndarray]:
    """Exact port of FuzzPuppy `eP`: 30ms smoothstep crossfade at M (or 15ms fade when overlap < 30ms)."""
    if audio_a is None and audio_b is None:
        return None
    n_channels = audio_a.shape[0] if audio_a is not None else audio_b.shape[0]
    total_samples = max(1, int(round(total_sec * sample_rate)))
    t = np.arange(total_samples, dtype=np.float64) / float(sample_rate)

    def _vec_smoothstep(x: np.ndarray) -> np.ndarray:
        c = np.clip(x, 0.0, 1.0)
        return c * c * (3.0 - 2.0 * c)

    if overlap_sec >= 0.03:
        w_b = _vec_smoothstep((t - m_sec + 0.015) / 0.03)
        w_a = 1.0 - w_b
    else:
        w_a = np.where(t < m_sec, _vec_smoothstep((m_sec - t) / 0.015), 0.0)
        w_b = np.where(t >= m_sec, _vec_smoothstep((t - m_sec) / 0.015), 0.0)

    out = np.zeros((n_channels, total_samples), dtype=np.float32)

    if audio_a is not None and audio_a.shape[1] > 0:
        mask_a = (w_a > 0.0) & (t < v_sec)
        idx_a = np.flatnonzero(mask_a)
        if idx_a.size > 0:
            src_pos = t[idx_a] * sample_rate
            valid_a = (src_pos >= 0.0) & (src_pos < audio_a.shape[1])
            idx_a = idx_a[valid_a]
            src_pos = src_pos[valid_a]
            if idx_a.size > 0:
                i0 = np.clip(np.floor(src_pos).astype(np.int64), 0, audio_a.shape[1] - 1)
                i1 = np.clip(i0 + 1, 0, audio_a.shape[1] - 1)
                frac = (src_pos - i0).astype(np.float32)
                a_ch = audio_a if audio_a.shape[0] == n_channels else np.repeat(audio_a[:1], n_channels, axis=0)
                samp_a = a_ch[:, i0] * (1.0 - frac) + a_ch[:, i1] * frac
                out[:, idx_a] += samp_a * w_a[idx_a].astype(np.float32)

    if audio_b is not None and audio_b.shape[1] > 0:
        mask_b = (w_b > 0.0) & (t >= b_sec)
        idx_b = np.flatnonzero(mask_b)
        if idx_b.size > 0:
            src_pos = (trim_b_sec + t[idx_b] - b_sec) * sample_rate
            valid_b = (src_pos >= 0.0) & (src_pos < audio_b.shape[1])
            idx_b = idx_b[valid_b]
            src_pos = src_pos[valid_b]
            if idx_b.size > 0:
                i0 = np.clip(np.floor(src_pos).astype(np.int64), 0, audio_b.shape[1] - 1)
                i1 = np.clip(i0 + 1, 0, audio_b.shape[1] - 1)
                frac = (src_pos - i0).astype(np.float32)
                b_ch = audio_b if audio_b.shape[0] == n_channels else np.repeat(audio_b[:1], n_channels, axis=0)
                samp_b = b_ch[:, i0] * (1.0 - frac) + b_ch[:, i1] * frac
                out[:, idx_b] += samp_b * w_b[idx_b].astype(np.float32)

    return out


def evaluate_seam_window_qc(
    frames_u8: np.ndarray,
    seam_start_idx: int,
    seam_span: int = 5,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Objective seam-window QC strictly scoped to `[seam_start-2 .. seam_start+span+2]` (never blames interior VFX)."""
    n = len(frames_u8)
    if n < 3:
        return {
            "qc_pass": True,
            "defect_score": 0.0,
            "peak_jump_mae": 0.0,
            "steady_mae": 0.01,
            "peak_jump_ratio": 1.0,
            "seam_luma_step": 0.0,
            "edge_drop_ratio": 0.0,
            "flags": [],
        }

    # Downsample only the local neighborhood around the seam for fast, deterministic QC
    lo = max(0, int(seam_start_idx) - 10)
    hi = min(n, int(seam_start_idx) + max(1, int(seam_span)) + 11)
    sub = frames_u8[lo:hi]
    local_seam_s = max(1, int(seam_start_idx) - lo)
    local_seam_e = min(len(sub) - 1, local_seam_s + max(1, int(seam_span)) + 2)
    win_lo = max(1, local_seam_s - 2)
    win_hi = max(win_lo + 1, local_seam_e)

    small = np.stack([_resize_u8(fr, 160, 90).astype(np.float32) / 255.0 for fr in sub], axis=0)
    step_maes = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2, 3))
    means = small.mean(axis=(1, 2))
    luma_steps = np.abs(np.diff(means, axis=0)).max(axis=1)

    seam_slice = slice(max(0, win_lo - 1), min(len(step_maes), win_hi))
    seam_maes = step_maes[seam_slice]
    seam_lumas = luma_steps[seam_slice]

    steady_indices = [i for i in range(len(step_maes)) if i < seam_slice.start or i >= seam_slice.stop]
    steady_mae = max(0.008, float(np.median(step_maes[steady_indices]))) if steady_indices else max(0.008, float(np.median(step_maes)))
    steady_luma = max(0.008, float(np.median(luma_steps[steady_indices]))) if steady_indices else 0.01

    peak_mae = float(seam_maes.max()) if seam_maes.size else 0.0
    peak_ratio = peak_mae / steady_mae
    peak_luma = float(seam_lumas.max()) if seam_lumas.size else 0.0

    # Measure Laplacian edge energy drop in the middle of the seam (detects severe double-ghosting / blur)
    ep_l = cv.cvtColor(sub[max(0, local_seam_s - 1)], cv.COLOR_RGB2GRAY).astype(np.float32)
    ep_r = cv.cvtColor(sub[min(len(sub) - 1, local_seam_s + max(1, int(seam_span)))], cv.COLOR_RGB2GRAY).astype(np.float32)
    ep_edge = 0.5 * (float(cv.Laplacian(ep_l, cv.CV_32F).var()) + float(cv.Laplacian(ep_r, cv.CV_32F).var()))
    mid_lo = min(len(sub) - 1, local_seam_s)
    mid_hi = min(len(sub), local_seam_s + max(1, int(seam_span)))
    if mid_hi > mid_lo and ep_edge > 15.0:
        mid_edges = [float(cv.Laplacian(cv.cvtColor(sub[k], cv.COLOR_RGB2GRAY).astype(np.float32), cv.CV_32F).var()) for k in range(mid_lo, mid_hi)]
        min_mid_edge = min(mid_edges)
        edge_drop_ratio = max(0.0, 1.0 - min_mid_edge / max(1e-3, ep_edge))
    else:
        edge_drop_ratio = 0.0

    flags = []
    if peak_ratio > 2.65 and peak_mae > 0.045:
        flags.append("seam_peak_jump")
    if peak_luma > max(0.08, steady_luma * 4.0):
        flags.append("seam_luma_step")
    if edge_drop_ratio > 0.58:
        flags.append("seam_ghost_blur")

    # Unified comparable defect score (lower is better; ~0.0-0.5 clean, >1.0 noticeable seam defect)
    defect_score = (
        max(0.0, peak_ratio - 1.35) * 0.55
        + peak_mae * 6.0
        + max(0.0, peak_luma - 0.03) * 8.0
        + max(0.0, edge_drop_ratio - 0.30) * 1.4
    )
    return {
        "qc_pass": len(flags) == 0,
        "defect_score": round(float(defect_score), 4),
        "peak_jump_mae": round(peak_mae, 5),
        "steady_mae": round(steady_mae, 5),
        "peak_jump_ratio": round(peak_ratio, 3),
        "seam_luma_step": round(peak_luma, 5),
        "edge_drop_ratio": round(edge_drop_ratio, 4),
        "flags": flags,
        "fps": round(float(fps), 3),
    }


def format_bilingual_summary(report: dict[str, Any]) -> tuple[str, str]:
    """Generate concise, truthful ZH and EN human-readable diagnostic summaries."""
    applied = str(report.get("applied") or "")
    err = report.get("structureError")
    cov = report.get("flowCoverage")
    backend = str(report.get("backend") or "none")
    qc = report.get("seam_qc") or {}
    qc_ok = qc.get("qc_pass", True)

    if applied == "manual_disabled_original":
        return "已手动关闭修复：原版零重编码直拼", "Repair disabled: lossless original concat"
    if applied == "passthrough_same_video":
        return "首段直通（无需拼接）", "First segment passthrough"
    if applied == "lossless_hard_cut":
        return "计划硬切：零重编码直拼", "Planned hard cut: lossless concat"
    if applied == "hard_cut_fallback":
        if report.get("reason") == "cut_candidate":
            return "自动兜底择优：原版硬切更干净", "Auto-fallback: original hard cut is cleaner"
        if err is not None and cov is not None:
            return (
                f"硬切保护回落原版 (结构差={err:.3f}, 覆盖率={cov:.3f})",
                f"Hard-cut protection fallback (err={err:.3f}, cov={cov:.3f})",
            )
        return "回落原版硬切", "Fallback to hard cut"

    qc_zh = "✅质检通过" if qc_ok else f"⚠接缝波动({','.join(qc.get('flags') or [])})"
    qc_en = "QC PASS" if qc_ok else f"QC WARN({','.join(qc.get('flags') or [])})"
    ladder_note_zh = " · 自动兜底择优" if report.get("ladder_switched") else ""
    ladder_note_en = " · auto-fallback picked" if report.get("ladder_switched") else ""
    return (
        f"已修复 [{applied} / {backend}]{ladder_note_zh} · 评分 {report.get('score', '-')} · {qc_zh}",
        f"Repaired [{applied} / {backend}]{ladder_note_en} · Score {report.get('score', '-')} · {qc_en}",
    )


def _attach_summaries(report: dict[str, Any]) -> dict[str, Any]:
    s_zh, s_en = format_bilingual_summary(report)
    report["summary_zh"] = s_zh
    report["summary_en"] = s_en
    return report


def _render_single_candidate(
    frames_a_u8: np.ndarray,
    frames_b_u8: np.ndarray,
    analysis: dict[str, Any],
    fps: float,
    mode: str,
    rife_bridge_frames: int,
    rife_backend: str,
    preserve_duration: bool,
    release_seconds: float,
    audio_a: Optional[np.ndarray],
    audio_b: Optional[np.ndarray],
    sample_rate: int,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Any]]:
    """Render one concrete candidate strategy (`rife`, `auto_align`, or `cut`) reusing precomputed `eI` analysis."""
    n_a, n_b = len(frames_a_u8), len(frames_b_u8)
    dur_a = analysis["dur_a"]
    dur_b = analysis["dur_b"]
    trim_a = analysis["trimA"]
    trim_b = analysis["trimB"]
    overlap = analysis["overlap"]
    v_sec = dur_a - trim_a
    b_sec = v_sec - overlap
    m_sec = v_sec - overlap / 2.0
    total_sec = v_sec + dur_b - trim_b - overlap
    out_count = max(2, int(round(total_sec * fps))) if preserve_duration else max(2, int(math.ceil(total_sec * fps - 1e-6)))

    report: dict[str, Any] = {
        "mode": mode,
        "preserve_duration": bool(preserve_duration),
        "safe": analysis["safe"],
        "structureError": analysis["structureError"],
        "flowCoverage": analysis["flowCoverage"],
        "motionDifference": analysis["motionDifference"],
        "colorDifference": analysis["colorDifference"],
        "score": analysis["score"],
        "quality": analysis["quality"],
        "trimA_frames": round(trim_a * fps, 2),
        "trimB_frames": round(trim_b * fps, 2),
        "overlap_frames": round(overlap * fps, 2),
        "input_a_frames": n_a,
        "input_b_frames": n_b,
    }

    if mode == "cut":
        out_frames = np.concatenate([frames_a_u8, frames_b_u8], axis=0)
        out_audio = merge_audio_eP(audio_a, audio_b, sample_rate, dur_a, dur_a, dur_a, 0.0, 0.0, dur_a + dur_b)
        qc = evaluate_seam_window_qc(out_frames, n_a, seam_span=2, fps=fps)
        report.update(
            applied="hard_cut_fallback",
            backend="none",
            output_frames=len(out_frames),
            frame_delta=0,
            seam_qc=qc,
            reason="cut_candidate",
        )
        return out_frames, out_audio, _attach_summaries(report)

    if mode == "rife":
        r_cnt = int(rife_bridge_frames)
        if r_cnt not in (3, 5, 7):
            r_cnt = 5

        if preserve_duration:
            left_cnt = r_cnt // 2
            right_cnt = r_cnt - left_cnt
            if n_a > left_cnt + 1 and n_b > right_cnt + 1:
                ep_a_idx = n_a - left_cnt - 1
                ep_b_idx = right_cnt
                mid_frames, backend_tag = run_rife_bridge(frames_a_u8[ep_a_idx], frames_b_u8[ep_b_idx], r_cnt, rife_backend)
                out_frames = np.concatenate([frames_a_u8[:ep_a_idx + 1], mid_frames, frames_b_u8[ep_b_idx:]], axis=0)
                out_audio = merge_audio_eP(audio_a, audio_b, sample_rate, dur_a, dur_a, dur_a, 0.0, 0.0, dur_a + dur_b)
                qc = evaluate_seam_window_qc(out_frames, ep_a_idx + 1, seam_span=r_cnt, fps=fps)
                report.update(
                    applied="rife_symmetric_zero_delta",
                    backend=backend_tag,
                    rife_left_a_idx=ep_a_idx,
                    rife_right_b_idx=ep_b_idx,
                    rife_bridge_frames=r_cnt,
                    output_frames=len(out_frames),
                    frame_delta=len(out_frames) - (n_a + n_b),
                    seam_qc=qc,
                    reason="repaired",
                )
                return out_frames, out_audio, _attach_summaries(report)

        max_frame_idx = out_count - 1
        if max_frame_idx >= r_cnt + 1:
            p_first = max(0, min(max_frame_idx - r_cnt - 1, int(round(m_sec * fps - (r_cnt + 1) / 2.0))))
            p_last = p_first + r_cnt + 1
            t_a_ep = p_first / fps
            t_b_ep = trim_b + p_last / fps - b_sec
            if t_a_ep < v_sec and 0.0 <= t_b_ep < dur_b:
                ep_a_idx, ep_a = _sample_frame_at_time(frames_a_u8, fps, t_a_ep)
                ep_b_idx, ep_b = _sample_frame_at_time(frames_b_u8, fps, t_b_ep)
                mid_frames, backend_tag = run_rife_bridge(ep_a, ep_b, r_cnt, rife_backend)

                out_list = []
                for f_idx in range(out_count):
                    t_out = f_idx / fps
                    if p_first < f_idx < p_last:
                        out_list.append(mid_frames[f_idx - p_first - 1])
                    elif f_idx <= p_first:
                        _, fr = _sample_frame_at_time(frames_a_u8, fps, min(t_out, max(0.0, dur_a - 1e-3)))
                        out_list.append(fr)
                    else:
                        t_b = min(trim_b + t_out - b_sec, max(0.0, dur_b - 1e-3))
                        _, fr = _sample_frame_at_time(frames_b_u8, fps, t_b)
                        out_list.append(fr)

                out_frames = np.stack(out_list, axis=0)
                out_audio = merge_audio_eP(
                    audio_a, audio_b, sample_rate, v_sec, b_sec, m_sec, trim_b, overlap, len(out_frames) / fps
                )
                qc = evaluate_seam_window_qc(out_frames, p_first + 1, seam_span=r_cnt, fps=fps)
                report.update(
                    applied="fuzzpuppy_rife",
                    backend=backend_tag,
                    rife_left_a_idx=ep_a_idx,
                    rife_right_b_idx=ep_b_idx,
                    rife_bridge_frames=r_cnt,
                    output_frames=len(out_frames),
                    frame_delta=len(out_frames) - (n_a + n_b),
                    seam_qc=qc,
                    reason="repaired",
                )
                return out_frames, out_audio, _attach_summaries(report)

        out_frames = np.concatenate([frames_a_u8, frames_b_u8], axis=0)
        out_audio = merge_audio_eP(audio_a, audio_b, sample_rate, dur_a, dur_a, dur_a, 0.0, 0.0, dur_a + dur_b)
        qc = evaluate_seam_window_qc(out_frames, n_a, seam_span=2, fps=fps)
        report.update(
            applied="hard_cut_fallback",
            backend="none",
            reason="clips_too_short_for_rife",
            output_frames=len(out_frames),
            frame_delta=0,
            seam_qc=qc,
        )
        return out_frames, out_audio, _attach_summaries(report)

    # Auto-align branch (`eS` + `eT`)
    subject_align = build_subject_alignment_eS(frames_a_u8, frames_b_u8, analysis, release_seconds=release_seconds)
    report["eS_coverage"] = round(subject_align["coverage"], 5)
    report["eS_residual"] = subject_align["residual"]
    report["release_seconds"] = round(subject_align["release"], 4)

    out_list = []
    tr_full = analysis["transform"]
    color = analysis["color"]
    align_w = subject_align["width"]
    rel_sec = subject_align["release"]

    for f_idx in range(out_count):
        t_out = f_idx / fps
        has_a = t_out < v_sec - 1e-6
        has_b = t_out >= b_sec - 1e-6

        if has_b and t_out < v_sec + rel_sec:
            _, raw_b = _sample_frame_at_time(frames_b_u8, fps, min(trim_b + t_out - b_sec, max(0.0, dur_b - 1e-3)))
            amount = _quintic_release(max(0.0, t_out - v_sec), rel_sec)
            flow_t = _interp_alignment_flow(subject_align, max(0.0, t_out - b_sec))
            warped_b = warp_composed_eT(raw_b, tr_full, flow_t, align_w, amount, color=color)
            if has_a and overlap > 1e-5:
                _, raw_a = _sample_frame_at_time(frames_a_u8, fps, min(t_out, max(0.0, dur_a - 1e-3)))
                alpha = _smoothstep((t_out - b_sec) / max(1.0 / fps, overlap - 1.0 / fps))
                out_list.append(blend_overlap_antighost(raw_a, warped_b, alpha))
            else:
                out_list.append(warped_b)
        elif has_b:
            _, raw_b = _sample_frame_at_time(frames_b_u8, fps, min(trim_b + t_out - b_sec, max(0.0, dur_b - 1e-3)))
            out_list.append(raw_b)
        else:
            _, raw_a = _sample_frame_at_time(frames_a_u8, fps, min(t_out, max(0.0, dur_a - 1e-3)))
            out_list.append(raw_a)

    out_frames = np.stack(out_list, axis=0)
    out_audio = merge_audio_eP(
        audio_a, audio_b, sample_rate, v_sec, b_sec, m_sec, trim_b, overlap, len(out_frames) / fps
    )
    seam_start = max(1, int(round(b_sec * fps)))
    seam_span = max(4, int(round(overlap * fps)) + 2)
    qc = evaluate_seam_window_qc(out_frames, seam_start, seam_span=seam_span, fps=fps)
    report.update(
        applied="fuzzpuppy_auto_align",
        backend="dense_lk_edge_diffuse_quintic",
        output_frames=len(out_frames),
        frame_delta=len(out_frames) - (n_a + n_b),
        seam_qc=qc,
        reason="repaired",
    )
    return out_frames, out_audio, _attach_summaries(report)


def merge_clips_fuzzpuppy(
    frames_a_u8: np.ndarray,
    frames_b_u8: np.ndarray,
    fps: float,
    mode: str = "rife",
    rife_bridge_frames: int = 5,
    rife_backend: str = "auto",
    preserve_duration: bool = False,
    max_search_sec: float = 0.5,
    overlap_frames: int = 4,
    structure_threshold: float = 0.38,
    coverage_threshold: float = 0.08,
    release_seconds: float = 0.8,
    audio_a: Optional[np.ndarray] = None,
    audio_b: Optional[np.ndarray] = None,
    sample_rate: int = 48000,
    retry_policy: str = "off",
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Any]]:
    """Run complete FuzzPuppy `eI + eS/eT + V + eP` merge with optional bounded 1-round orthogonal fallback (`once_ladder`)."""
    frames_a_u8 = _ensure_u8_rgb(frames_a_u8)
    frames_b_u8 = _ensure_u8_rgb(frames_b_u8)
    if len(frames_a_u8) == 0 or len(frames_b_u8) == 0:
        raise ValueError("merge_clips_fuzzpuppy requires non-empty Clip A and Clip B frames.")
    h, w = frames_a_u8.shape[1:3]
    if frames_b_u8.shape[1:3] != (h, w):
        frames_b_u8 = np.stack([_fit_frame_to_size(fr, w, h) for fr in frames_b_u8], axis=0)

    fps = float(_clamp(float(fps), 12.0, 60.0))
    n_a, n_b = len(frames_a_u8), len(frames_b_u8)

    if mode == "cut":
        out_frames = np.concatenate([frames_a_u8, frames_b_u8], axis=0)
        out_audio = None
        if audio_a is not None or audio_b is not None:
            dur_a = n_a / fps
            dur_b = n_b / fps
            out_audio = merge_audio_eP(audio_a, audio_b, sample_rate, dur_a, dur_a, dur_a, 0.0, 0.0, dur_a + dur_b)
        qc = evaluate_seam_window_qc(out_frames, n_a, seam_span=2, fps=fps)
        return out_frames, out_audio, _attach_summaries({
            "mode": "cut",
            "applied": "lossless_hard_cut",
            "safe": False,
            "reason": "explicit_cut",
            "input_a_frames": n_a,
            "input_b_frames": n_b,
            "output_frames": len(out_frames),
            "frame_delta": 0,
            "seam_qc": qc,
        })

    analysis = analyze_seam_eI(
        frames_a_u8,
        frames_b_u8,
        fps=fps,
        max_search_sec=max_search_sec,
        overlap_frames=overlap_frames,
        structure_threshold=structure_threshold,
        coverage_threshold=coverage_threshold,
        preserve_duration=preserve_duration,
    )

    if not analysis["safe"]:
        out_frames = np.concatenate([frames_a_u8, frames_b_u8], axis=0)
        dur_a = n_a / fps
        dur_b = n_b / fps
        out_audio = merge_audio_eP(audio_a, audio_b, sample_rate, dur_a, dur_a, dur_a, 0.0, 0.0, dur_a + dur_b)
        qc = evaluate_seam_window_qc(out_frames, n_a, seam_span=2, fps=fps)
        return out_frames, out_audio, _attach_summaries({
            "mode": mode,
            "preserve_duration": bool(preserve_duration),
            "safe": False,
            "structureError": analysis["structureError"],
            "flowCoverage": analysis["flowCoverage"],
            "motionDifference": analysis["motionDifference"],
            "colorDifference": analysis["colorDifference"],
            "score": analysis["score"],
            "quality": analysis["quality"],
            "trimA_frames": 0.0,
            "trimB_frames": 0.0,
            "overlap_frames": 0.0,
            "input_a_frames": n_a,
            "input_b_frames": n_b,
            "applied": "hard_cut_fallback",
            "backend": "none",
            "reason": "correspondence_unreliable_fallback_to_cut",
            "output_frames": len(out_frames),
            "frame_delta": 0,
            "seam_qc": qc,
        })

    # Step 1: Primary candidate
    best_frames, best_audio, best_rep = _render_single_candidate(
        frames_a_u8, frames_b_u8, analysis, fps, mode, rife_bridge_frames, rife_backend,
        preserve_duration, release_seconds, audio_a, audio_b, sample_rate,
    )
    best_rep["retry_policy"] = retry_policy
    best_rep["ladder_switched"] = False

    # Step 2 (Bounded 1-round orthogonal fallback ONLY when retry_policy == "once_ladder" and primary QC warns)
    if retry_policy == "once_ladder" and not best_rep.get("seam_qc", {}).get("qc_pass", True):
        primary_tag = f"{mode}:{rife_bridge_frames}" if mode == "rife" else mode
        if mode == "rife":
            alt_specs = [("auto_align", rife_bridge_frames)]
            alt_span = 7 if int(rife_bridge_frames) != 7 else 5
            alt_specs.append(("rife", alt_span))
            alt_specs.append(("cut", 0))
        else:
            alt_specs = [("rife", 5), ("rife", 7), ("cut", 0)]

        trials = [{
            "candidate": primary_tag,
            "defect_score": best_rep["seam_qc"]["defect_score"],
            "qc_pass": best_rep["seam_qc"]["qc_pass"],
            "flags": best_rep["seam_qc"]["flags"],
        }]
        best_score = float(best_rep["seam_qc"]["defect_score"])
        winner_tag = primary_tag

        for cand_mode, cand_span in alt_specs:
            try:
                c_frames, c_audio, c_rep = _render_single_candidate(
                    frames_a_u8, frames_b_u8, analysis, fps, cand_mode, cand_span, rife_backend,
                    preserve_duration, release_seconds, audio_a, audio_b, sample_rate,
                )
            except Exception:
                continue
            c_tag = f"{cand_mode}:{cand_span}" if cand_mode == "rife" else cand_mode
            c_qc = c_rep["seam_qc"]
            c_score = float(c_qc["defect_score"])
            trials.append({
                "candidate": c_tag,
                "defect_score": c_qc["defect_score"],
                "qc_pass": c_qc["qc_pass"],
                "flags": c_qc["flags"],
            })
            # Prefer strictly passing or meaningfully lower defect score (avoid switching on noise < 0.05)
            better_pass = bool(c_qc["qc_pass"]) and not bool(best_rep["seam_qc"]["qc_pass"])
            if better_pass or (c_score + 0.05 < best_score):
                best_frames, best_audio, best_rep = c_frames, c_audio, c_rep
                best_score = c_score
                winner_tag = c_tag

        best_rep["retry_policy"] = retry_policy
        best_rep["ladder_switched"] = (winner_tag != primary_tag)
        best_rep["ladder_trials"] = trials
        best_rep = _attach_summaries(best_rep)

    return best_frames, best_audio, best_rep
