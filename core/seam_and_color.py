"""Zero-jump 12-frame tapered MKL-Lab seam matcher, masked background YCbCr matcher, and exact AV trim."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..compat import resample_audio

_RGB_TO_YCC = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=torch.float32,
)
_YCC_TO_RGB = torch.tensor(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=torch.float32,
)
_SRGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float32,
)
_XYZ_TO_SRGB = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=torch.float32,
)
_D65_WHITE = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)


def _srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _SRGB_TO_XYZ.to(rgb).T
    xyz = xyz / _D65_WHITE.to(rgb)
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = torch.where(xyz > eps, xyz.clamp(min=1e-8) ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack((L, a, b), dim=-1)


def _lab_to_srgb(lab: torch.Tensor) -> torch.Tensor:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps_cbrt = (216.0 / 24389.0) ** (1.0 / 3.0)
    kappa = 24389.0 / 27.0
    xr = torch.where(fx > eps_cbrt, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = torch.where(L > (kappa * (216.0 / 24389.0)), fy ** 3, L / kappa)
    zr = torch.where(fz > eps_cbrt, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = torch.stack((xr, yr, zr), dim=-1) * _D65_WHITE.to(lab)
    lin = (xyz @ _XYZ_TO_SRGB.to(lab).T).clamp(0.0, 1.0)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * (lin.clamp(min=1e-8) ** (1.0 / 2.4)) - 0.055).clamp(0.0, 1.0)


def _parse_strength_curve(curve_str: str, length: int) -> list[float]:
    anchors = []
    for part in str(curve_str or "0.75, 0.45, 0.18").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            anchors.append(max(0.0, min(1.0, float(part))))
        except ValueError:
            pass
    if not anchors:
        anchors = [0.75, 0.45, 0.18]
    anchors.append(0.0)
    if length <= 1:
        return [anchors[0]]
    pts = torch.tensor(anchors, dtype=torch.float32).view(1, 1, -1)
    sampled = F.interpolate(pts, size=length + 1, mode="linear", align_corners=True).view(-1)
    return [float(x) for x in sampled[:length].tolist()]


def _mkl_transform_matrix(src_pixels: torch.Tensor, ref_pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute MKL color covariance transfer matrix in CIE Lab space on GPU/CPU."""
    mu_s = src_pixels.mean(0, keepdim=True)
    mu_r = ref_pixels.mean(0, keepdim=True)
    xs = src_pixels - mu_s
    xr = ref_pixels - mu_r
    cov_s = (xs.T @ xs) / max(1, xs.shape[0] - 1) + 1e-4 * torch.eye(3, device=xs.device)
    cov_r = (xr.T @ xr) / max(1, xr.shape[0] - 1) + 1e-4 * torch.eye(3, device=xr.device)
    eigval_s, eigvec_s = torch.linalg.eigh(cov_s)
    eigval_s = eigval_s.clamp(min=1e-6)
    s_half = eigvec_s @ torch.diag(eigval_s.sqrt()) @ eigvec_s.T
    s_inv_half = eigvec_s @ torch.diag(eigval_s.rsqrt()) @ eigvec_s.T
    mid = s_half @ cov_r @ s_half
    eigval_m, eigvec_m = torch.linalg.eigh(mid)
    eigval_m = eigval_m.clamp(min=1e-6)
    mid_half = eigvec_m @ torch.diag(eigval_m.sqrt()) @ eigvec_m.T
    T = s_inv_half @ mid_half @ s_inv_half
    return T, mu_s, mu_r


def mkl_lab_seam_match(
    images: torch.Tensor,
    reference_clean_tail: Optional[torch.Tensor],
    seam_frames: int = 12,
    strength_curve: str = "0.75, 0.45, 0.18",
) -> torch.Tensor:
    """Apply smooth fading CIE-Lab MKL color alignment on the first `min(seam_frames, N)` frames."""
    if reference_clean_tail is None or not isinstance(reference_clean_tail, torch.Tensor):
        return images
    if images.ndim != 4 or images.shape[0] == 0 or reference_clean_tail.ndim != 4 or reference_clean_tail.shape[0] == 0:
        return images
    n_seam = min(max(1, int(seam_frames)), int(images.shape[0]))
    ref = reference_clean_tail[-min(4, int(reference_clean_tail.shape[0])):, ..., :3].to(device=images.device, dtype=torch.float32)
    head = images[:n_seam, ..., :3].float()
    step = max(1, int(math.sqrt(head.shape[1] * head.shape[2] / 16384)))
    src_sub = _srgb_to_lab(head[:min(3, n_seam), ::step, ::step, :].reshape(-1, 3))
    ref_sub = _srgb_to_lab(ref[:, ::step, ::step, :].reshape(-1, 3))
    T, mu_s, mu_r = _mkl_transform_matrix(src_sub, ref_sub)
    weights = _parse_strength_curve(strength_curve, n_seam)

    out = images.clone()
    head_lab = _srgb_to_lab(head)
    matched_lab = ((head_lab.reshape(-1, 3) - mu_s) @ T + mu_r).reshape_as(head_lab)
    w_tensor = torch.tensor(weights, device=images.device, dtype=torch.float32).view(n_seam, 1, 1, 1)
    blended_rgb = _lab_to_srgb(head_lab * (1.0 - w_tensor) + matched_lab * w_tensor)
    out[:n_seam, ..., :3] = blended_rgb.to(dtype=images.dtype)
    return out


def _rgb_to_lab_exact(rgb_bhwc: torch.Tensor) -> torch.Tensor:
    """Convert `[B, H, W, 3]` float32 sRGB `(0..1)` to `[B, 3, H, W]` CIE Lab via Kornia (fallback to internal)."""
    x = rgb_bhwc[..., :3].float().clamp(0.0, 1.0)
    try:
        import kornia.color  # type: ignore

        return kornia.color.rgb_to_lab(x.permute(0, 3, 1, 2).contiguous())
    except Exception:
        return _srgb_to_lab(x).permute(0, 3, 1, 2).contiguous()


def _lab_to_rgb_exact(lab_bchw: torch.Tensor) -> torch.Tensor:
    """Convert `[B, 3, H, W]` CIE Lab back to `[B, H, W, 3]` float32 sRGB `(0..1)`."""
    try:
        import kornia.color  # type: ignore

        return kornia.color.lab_to_rgb(lab_bchw).permute(0, 2, 3, 1).clamp(0.0, 1.0).contiguous()
    except Exception:
        return _lab_to_srgb(lab_bchw.permute(0, 2, 3, 1)).clamp(0.0, 1.0).contiguous()


def _parse_initial_strength(curve_or_float: Any, default: float = 0.88) -> float:
    """Extract initial seam strength `s0 in [0.0, 1.0]` from `'0.88'` or legacy `'0.75, 0.45, 0.18'`."""
    if isinstance(curve_or_float, (int, float)):
        return max(0.0, min(1.0, float(curve_or_float)))
    for part in str(curve_or_float or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            return max(0.0, min(1.0, float(part)))
        except ValueError:
            continue
    return max(0.0, min(1.0, float(default)))


def _pick_first_clean_head_index(head_rgb: torch.Tensor, ref_last_rgb: torch.Tensor, max_scan: int = 3) -> int:
    """Guard against corrupted 1..3-frame white/black flash at `curr[0]` so MKL baseline uses first clean frame."""
    n = int(head_rgb.shape[0])
    if n <= 1:
        return 0
    lim = min(max_scan, n - 1)
    luma_ref = float((0.2126 * ref_last_rgb[..., 0] + 0.7152 * ref_last_rgb[..., 1] + 0.0722 * ref_last_rgb[..., 2]).mean().item())
    luma_head = 0.2126 * head_rgb[:lim + 2, ..., 0] + 0.7152 * head_rgb[:lim + 2, ..., 1] + 0.0722 * head_rgb[:lim + 2, ..., 2]
    means = luma_head.mean(dim=(1, 2))
    post_ref = float(means[min(lim, means.shape[0] - 1):].median().item())
    hi_ref = max(luma_ref, post_ref)
    lo_ref = min(luma_ref, post_ref)
    for idx in range(lim):
        m = float(means[idx].item())
        clip_hi = float((luma_head[idx] > 0.94).float().mean().item())
        clip_lo = float((luma_head[idx] < 0.03).float().mean().item())
        is_bad_white = (clip_hi > 0.35 and m > hi_ref + 0.09) or (m > hi_ref + 0.16)
        is_bad_black = clip_lo > 0.65 and m < lo_ref - 0.14
        if not (is_bad_white or is_bad_black):
            return idx
    return lim


def linear_22f_mkl_lab_match(
    images: torch.Tensor,
    reference_clean_tail: Optional[torch.Tensor],
    seam_frames: int = 22,
    initial_strength: Any = 0.88,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Dense arithmetic-progression linear MKL-Lab decay across `seam_frames` (default 22f) anchored strictly to `prev[-1:]`.

    Weight schedule: `w_i = s0 * (N - i) / N` for `i = 0 .. N-1`, `w_N = 0.0`, giving a constant per-frame step
    `delta_w = s0 / N` all the way to untouched frame `N` with zero staircase jumps and zero VFX suppression.
    """
    rep: dict[str, Any] = {"mode": "linear_22f_mkl_lab", "applied_color": False}
    if reference_clean_tail is None or not isinstance(reference_clean_tail, torch.Tensor):
        rep["reason"] = "no_reference_tail"
        return images, rep
    if images.ndim != 4 or int(images.shape[0]) == 0 or reference_clean_tail.ndim != 4 or int(reference_clean_tail.shape[0]) == 0:
        rep["reason"] = "empty_input"
        return images, rep

    s0 = _parse_initial_strength(initial_strength, default=0.88)
    n_total = int(images.shape[0])
    n_seam = min(max(1, int(seam_frames)), n_total)
    if s0 <= 0.0:
        rep["reason"] = "zero_strength"
        return images, rep

    dev = images.device
    ref_last = reference_clean_tail[-1:, ..., :3].to(device=dev, dtype=torch.float32)
    head_slice = images[:min(n_seam, max(6, n_seam)), ..., :3].float()
    anchor_idx = _pick_first_clean_head_index(head_slice, ref_last[0], max_scan=min(3, n_seam - 1))

    ref_lab = _rgb_to_lab_exact(ref_last).view(3, -1)
    src_lab = _rgb_to_lab_exact(images[anchor_idx:anchor_idx + 1, ..., :3].float()).view(3, -1)

    eps = 1e-5
    mu_r = ref_lab.mean(dim=-1, keepdim=True)
    mu_s = src_lab.mean(dim=-1, keepdim=True)
    cr = ref_lab - mu_r
    cs = src_lab - mu_s
    cov_r = (cr @ cr.T) / max(1, cr.shape[1]) + eps * torch.eye(3, device=dev, dtype=torch.float32)
    cov_s = (cs @ cs.T) / max(1, cs.shape[1]) + eps * torch.eye(3, device=dev, dtype=torch.float32)

    eig_s, vec_s = torch.linalg.eigh(cov_s)
    sqrt_s = torch.sqrt(eig_s.clamp_min(eps))
    scaled_v = vec_s * sqrt_s.unsqueeze(0)
    mid = scaled_v.T @ cov_r @ scaled_v
    eig_m, vec_m = torch.linalg.eigh(mid)
    sqrt_m = torch.sqrt(eig_m.clamp_min(0.0))
    inv_scaled_v = vec_s * (1.0 / sqrt_s).unsqueeze(0)
    m_half = (vec_m * sqrt_m.unsqueeze(0)) @ vec_m.T
    T_raw = inv_scaled_v @ m_half @ inv_scaled_v.T

    u_svd, s_svd, vh_svd = torch.linalg.svd(T_raw)
    T = u_svd @ torch.diag(s_svd.clamp(0.60, 1.65)) @ vh_svd
    offset = mu_r - T @ mu_s

    de_before = float(torch.norm(mu_s.squeeze(-1) - mu_r.squeeze(-1)).item())

    # Exact arithmetic progression: w_i = s0 * (n_seam - i) / n_seam, constant step = s0 / n_seam -> 0 at i = n_seam
    weights = [s0 * float(n_seam - i) / float(n_seam) for i in range(n_seam)]
    out = images.clone()
    h_px, w_px = int(images.shape[1]), int(images.shape[2])
    first_after_mean = mu_s
    for i, w_i in enumerate(weights):
        f_lab = _rgb_to_lab_exact(images[i:i + 1, ..., :3].float())  # [1, 3, H, W]
        corr = (T @ f_lab.view(3, -1) + offset).view(1, 3, h_px, w_px)
        blended_lab = corr if w_i >= 1.0 - 1e-6 else torch.lerp(f_lab, corr, w_i)
        if i == anchor_idx:
            first_after_mean = blended_lab.view(3, -1).mean(dim=-1, keepdim=True)
        out[i:i + 1, ..., :3] = _lab_to_rgb_exact(blended_lab).to(dtype=images.dtype)

    de_after = float(torch.norm(first_after_mean.squeeze(-1) - mu_r.squeeze(-1)).item())
    rep.update({
        "applied_color": True,
        "seam_frames": n_seam,
        "initial_strength": round(s0, 4),
        "step_delta": round(s0 / float(n_seam), 6),
        "anchor_idx": int(anchor_idx),
        "delta_e_seam": round(de_before, 3),
        "delta_e_after": round(de_after, 3),
    })
    return out, rep


def _shot_bounds(reference_chw: torch.Tensor) -> list[tuple[int, int]]:
    count = int(reference_chw.shape[0])
    if count < 2:
        return [(0, count)]
    bins = 8
    levels = (reference_chw.clamp(0.0, 0.999) * bins).long()
    idx = (levels[:, 0] * bins + levels[:, 1]) * bins + levels[:, 2]
    hist = torch.stack([torch.bincount(f.flatten(), minlength=bins ** 3) for f in idx]).float()
    hist = hist / hist.sum(1, keepdim=True).clamp(min=1e-6)
    jumps = [0.5 * float((hist[i] - hist[i + 1]).abs().sum()) for i in range(count - 1)]
    threshold = max(0.5, 3.0 * float(torch.tensor(jumps).median()))
    cuts = []
    for i, jump in enumerate(jumps):
        if jump < threshold:
            continue
        if i + 2 < count and 0.5 * float((hist[i] - hist[i + 2]).abs().sum()) < 0.5 * jump:
            continue
        if i >= 1 and 0.5 * float((hist[i - 1] - hist[i + 1]).abs().sum()) < 0.5 * jump:
            continue
        cuts.append(i + 1)
    bounds = list(zip([0] + cuts, cuts + [count]))
    while len(bounds) > 1 and min(e - s for s, e in bounds) < 6:
        short = min(range(len(bounds)), key=lambda i: bounds[i][1] - bounds[i][0])
        if short == 0:
            nbr = 1
        elif short == len(bounds) - 1:
            nbr = short - 1
        else:
            b0 = bounds[short - 1][1] - bounds[short - 1][0]
            b1 = bounds[short + 1][1] - bounds[short + 1][0]
            nbr = short - 1 if b0 >= b1 else short + 1
        i0, i1 = sorted((short, nbr))
        bounds[i0:i1 + 1] = [(bounds[i0][0], bounds[i1][1])]
    return bounds


def masked_bg_ycc_match(
    images: torch.Tensor,
    source_bg_frames: Optional[torch.Tensor],
    strength: float = 1.0,
    min_background_share: float = 0.20,
    max_typical_diff: float = 0.12,
) -> tuple[torch.Tensor, list[dict]]:
    """Correct global YCbCr drift per detected shot using only unchanged background pixels."""
    strength = max(0.0, min(1.0, float(strength)))
    if source_bg_frames is None or strength == 0.0 or int(images.shape[0]) == 0 or int(source_bg_frames.shape[0]) == 0:
        return images, []
    count = min(int(images.shape[0]), int(source_bg_frames.shape[0]))
    ah = 120
    aw = max(8, round(ah * float(images.shape[2]) / max(1, int(images.shape[1]))))
    gen_chw = F.interpolate(images[:count, ..., :3].float().movedim(-1, 1), size=(ah, aw), mode="area")
    src_chw = F.interpolate(source_bg_frames[:count, ..., :3].float().to(images.device).movedim(-1, 1), size=(ah, aw), mode="area")

    def _blur(c):
        return F.avg_pool2d(c, 3, 1, 1, count_include_pad=False)

    diff = (_blur(gen_chw) - _blur(src_chw)).abs().mean(1, keepdim=True)
    typical = diff.flatten(1).median(1).values
    changed = diff > (typical * 3.0).clamp(min=0.06).view(-1, 1, 1, 1)
    bg_mask = ~(F.max_pool2d(changed.float(), 5, 1, 2) > 0)

    rgb_to_ycc = _RGB_TO_YCC.to(device=images.device)
    ycc_to_rgb = _YCC_TO_RGB.to(device=images.device)
    out = images.clone()
    report: list[dict] = []
    for s, e in _shot_bounds(src_chw):
        mask = bg_mask[s:e, 0]
        share = float(mask.float().mean())
        med_diff = float(typical[s:e].median())
        entry = {"frames": (s, e), "background": round(share, 3), "median_diff": round(med_diff, 4), "applied": False}
        report.append(entry)
        if share < min_background_share or med_diff > max_typical_diff:
            continue
        ours = gen_chw[s:e].movedim(1, -1)[mask] @ rgb_to_ycc.T
        theirs = src_chw[s:e].movedim(1, -1)[mask] @ rgb_to_ycc.T
        mu_s, std_s = ours.mean(0), ours.std(0).clamp(min=1e-4)
        mu_r, std_r = theirs.mean(0), theirs.std(0).clamp(min=1e-4)
        scale = (std_r / std_s).clamp(0.67, 1.50)
        for c_start in range(s, e, 16):
            c_end = min(e, c_start + 16)
            chunk = images[c_start:c_end, ..., :3].float()
            ycc = chunk @ rgb_to_ycc.T
            matched = (ycc - mu_s) * scale + mu_r
            ycc = ycc + strength * (matched - ycc)
            out[c_start:c_end, ..., :3] = (ycc @ ycc_to_rgb.T).clamp(0.0, 1.0).to(dtype=images.dtype)
        entry["applied"] = True
    return out, report


def trim_av_sync(
    images: torch.Tensor,
    audio: Optional[dict],
    trim_head_frames: int = 0,
    keep_frames: int = 0,
    fps: int = 24,
) -> tuple[torch.Tensor, Optional[dict]]:
    """Trim lead-in overlap frames and optional tail overflow (`keep_frames > 0`) symmetrically across image & audio."""
    total = int(images.shape[0])
    head = max(0, min(int(trim_head_frames), max(0, total - 1)))
    if int(keep_frames) > 0:
        end = min(total, head + int(keep_frames))
    else:
        end = total
    trimmed_images = images[head:end]
    kept = int(trimmed_images.shape[0])

    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        return trimmed_images, audio

    wav = audio["waveform"]
    sr = int(audio.get("sample_rate", 48000))
    if sr != 48000:
        wav = resample_audio(wav, sr, 48000)
        sr = 48000
    s0 = round(head / float(fps) * sr)
    target_len = max(1, round(kept / float(fps) * sr))
    sliced = wav[..., s0:s0 + target_len]
    if sliced.shape[-1] < target_len:
        sliced = F.pad(sliced, (0, target_len - sliced.shape[-1]))
    return trimmed_images, {"waveform": sliced.contiguous(), "sample_rate": sr}


# ==============================================================================
# v5.3 GPU 3-Gate Seam Drift Detection, Dual-Layer Bounded MKL-Lab & O(1) Heal
# ==============================================================================

def _bt709_norm_luma(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Zero-mean normalized BT.709 luminance `[B, H, W]` with std floored at 0.08."""
    luma = 0.2126 * rgb_chw[:, 0] + 0.7152 * rgb_chw[:, 1] + 0.0722 * rgb_chw[:, 2]
    mean = luma.mean(dim=(1, 2), keepdim=True)
    std = luma.var(dim=(1, 2), unbiased=False, keepdim=True).sqrt().clamp(min=0.08)
    return (luma - mean) / std


def _register_translation_128(la: torch.Tensor, lb: torch.Tensor) -> tuple[int, int, float]:
    """Vectorized GPU/CPU coarse+fine translation registration on `[H_s, W_s]` normalized luma."""
    h, w = int(la.shape[0]), int(la.shape[1])
    if h < 28 or w < 44:
        return 0, 0, float((la - lb).abs().clamp(max=1.0).mean().item())
    my, mx = 10, 16
    ref = la[my:h - my, mx:w - mx]
    best_dx, best_dy, best_err = 0, 0, 9.9
    for dy in range(-10, 11, 2):
        for dx in range(-16, 17, 2):
            err = float((ref - lb[my + dy:h - my + dy, mx + dx:w - mx + dx]).abs().clamp(max=1.0).mean().item())
            if err < best_err:
                best_err, best_dx, best_dy = err, dx, dy
    cx, cy = best_dx, best_dy
    for dy in range(max(-10, cy - 1), min(11, cy + 2)):
        for dx in range(max(-16, cx - 1), min(17, cx + 2)):
            err = float((ref - lb[my + dy:h - my + dy, mx + dx:w - mx + dx]).abs().clamp(max=1.0).mean().item())
            if err < best_err:
                best_err, best_dx, best_dy = err, dx, dy
    return best_dx, best_dy, float(best_err)


def _batch_patch_match_128(fa: torch.Tensor, fb: torch.Tensor, px: torch.Tensor, py: torch.Tensor):
    """Vectorized PyTorch 16-tap 7x7 patch match across 121 shifts (-5..5) for P centers."""
    p_cnt = int(px.numel())
    if p_cnt == 0:
        z = torch.zeros((0,), dtype=torch.float32, device=fa.device)
        return z.long(), z.long(), z
    h, w = int(fa.shape[0]), int(fa.shape[1])
    dev = fa.device
    g1d = torch.arange(-3, 4, device=dev)
    gx, gy = torch.meshgrid(g1d, g1d, indexing="xy")
    xf = gx.reshape(1, -1)
    yf = gy.reshape(1, -1)
    xr = (px[:, None] + xf + 1).clamp(0, w - 1)
    xl = (px[:, None] + xf).clamp(0, w - 1)
    yt = (py[:, None] + yf).clamp(0, h - 1)
    s_tex = (fa[yt, xr] - fa[yt, xl]).abs().sum(dim=1)

    taps = torch.tensor([-3, -1, 1, 3], device=dev)
    ox_g, oy_g = torch.meshgrid(taps, taps, indexing="xy")
    ox = ox_g.reshape(1, 16)
    oy = oy_g.reshape(1, 16)
    pa = fa[(py[:, None] + oy).clamp(0, h - 1), (px[:, None] + ox).clamp(0, w - 1)]

    s1d = torch.arange(-5, 6, device=dev)
    sy_g, sx_g = torch.meshgrid(s1d, s1d, indexing="ij")
    sx_f, sy_f = sx_g.reshape(1, 121, 1), sy_g.reshape(1, 121, 1)
    qx = (px[:, None, None] + ox[:, None, :] + sx_f).clamp(0, w - 1)
    qy = (py[:, None, None] + oy[:, None, :] + sy_f).clamp(0, h - 1)
    pb = fb[qy, qx]

    d = (pa[:, None, :] - pb).abs().mean(dim=2)
    b1, idx = d.min(dim=1)
    d2 = d.clone()
    d2[torch.arange(p_cnt, device=dev), idx] = 1e9
    b2 = d2.min(dim=1).values
    dx = sx_f.reshape(-1)[idx]
    dy = sy_f.reshape(-1)[idx]
    conf = (
        ((b2 - b1) / b2.clamp(min=0.02) * 3.0).clamp(0.0, 1.0)
        * (s_tex / 8.0).clamp(0.0, 1.0)
        * (1.0 - b1 / 0.55).clamp(0.0, 1.0)
    )
    return dx, dy, conf


def _bidirectional_patch_coverage(la: torch.Tensor, lb: torch.Tensor) -> float:
    """Bidirectional 7x7 patch consistency ratio across an 8px grid on 128px luma."""
    h, w = int(la.shape[0]), int(la.shape[1])
    gw, gh = max(1, math.ceil(w / 8.0)), max(1, math.ceil(h / 8.0))
    xs = torch.arange(4, w, 8, device=la.device)
    ys = torch.arange(4, h, 8, device=la.device)
    if xs.numel() == 0 or ys.numel() == 0:
        return 0.0
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    px, py = gx.reshape(-1), gy.reshape(-1)
    keep = (px >= 9) & (px < w - 9) & (py >= 9) & (py < h - 9)
    px, py = px[keep], py[keep]
    if px.numel() == 0:
        return 0.0
    dx1, dy1, conf1 = _batch_patch_match_128(la, lb, px, py)
    dx2, dy2, _ = _batch_patch_match_128(lb, la, px + dx1, py + dy1)
    fb = torch.hypot((dx1 + dx2).float(), (dy1 + dy2).float())
    valid = (fb <= 1.5) & (dx1.abs() < 5) & (dy1.abs() < 5) & (conf1 > 0.15)
    return float(valid.sum().item()) / float(max(1, gw * gh))


def _shift_align_pair(ref_chw: torch.Tensor, curr_chw: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop overlapping `[C, H', W']` view between reference and translated current frame."""
    h, w = int(ref_chw.shape[1]), int(ref_chw.shape[2])
    my, mx = max(0, abs(dy)), max(0, abs(dx))
    if h - 2 * my < 8 or w - 2 * mx < 8:
        return ref_chw, curr_chw
    return (
        ref_chw[:, my:h - my, mx:w - mx],
        curr_chw[:, my + dy:h - my + dy, mx + dx:w - mx + dx],
    )


def _compute_static_bg_mask(ref_chw: torch.Tensor, curr_chw: torch.Tensor) -> torch.Tensor:
    """Exclude localized VFX/moving foreground after stripping global median color offset."""
    rb = F.avg_pool2d(ref_chw.unsqueeze(0), 3, 1, 1, count_include_pad=False)
    cb = F.avg_pool2d(curr_chw.unsqueeze(0), 3, 1, 1, count_include_pad=False)
    signed = cb - rb
    med_offset = signed.flatten(2).median(dim=2).values.view(1, -1, 1, 1)
    resid = (signed - med_offset).abs().mean(dim=1, keepdim=True)
    typical = float(resid.median().item())
    changed = resid > max(0.06, 3.0 * typical)
    static_mask = ~(F.max_pool2d(changed.float(), 5, 1, 2) > 0)[0, 0]
    non_black = ref_chw.sum(dim=0) >= (12.0 / 255.0)
    mask = static_mask & non_black
    if float(mask.float().mean().item()) >= 0.15:
        return mask
    return non_black if float(non_black.float().mean().item()) >= 0.15 else torch.ones_like(mask, dtype=torch.bool)


def detect_seam_gate_and_drift(
    curr_images: torch.Tensor,
    ref_tail: Optional[torch.Tensor],
    scene_anchor: Optional[torch.Tensor] = None,
    deadzone_de: float = 1.5,
    known_continuation: bool = False,
) -> dict[str, Any]:
    """GPU 3-gate detector: same-scene patch gate, flash immunity, static bg mask, and Lab ΔE deadzone."""
    out: dict[str, Any] = {
        "same_scene_safe": False, "is_flash": False, "luma_err": 1.0, "coverage": 0.0, "hist_l1": 1.0,
        "dx": 0, "dy": 0, "delta_e_seam": 0.0, "delta_e_anchor": 0.0, "delta_e_eff": 0.0,
        "luma_std_ratio": 1.0, "static_share": 0.0, "alpha_auto": 0.0, "reason": "no_ref",
    }
    if ref_tail is None or not isinstance(ref_tail, torch.Tensor) or curr_images.ndim != 4 or int(curr_images.shape[0]) == 0 or int(ref_tail.shape[0]) == 0:
        return out

    h, w = int(curr_images.shape[1]), int(curr_images.shape[2])
    h_s = max(24, round(h * 128.0 / max(1, w)))
    dev = curr_images.device
    c_sub = F.interpolate(curr_images[:min(8, int(curr_images.shape[0])), ..., :3].float().movedim(-1, 1), size=(h_s, 128), mode="area")
    r_sub = F.interpolate(ref_tail[-min(4, int(ref_tail.shape[0])):, ..., :3].to(device=dev).float().movedim(-1, 1), size=(h_s, 128), mode="area")

    la = _bt709_norm_luma(r_sub[-1:])[0]
    lb = _bt709_norm_luma(c_sub[:1])[0]
    raw_ya = (0.2126 * r_sub[-1, 0] + 0.7152 * r_sub[-1, 1] + 0.0722 * r_sub[-1, 2])
    raw_yb = (0.2126 * c_sub[0, 0] + 0.7152 * c_sub[0, 1] + 0.0722 * c_sub[0, 2])
    std_a, std_b = max(1e-4, float(raw_ya.std().item())), max(1e-4, float(raw_yb.std().item()))

    dx, dy, luma_err = _register_translation_128(la, lb)
    ra_crop, cb_crop = _shift_align_pair(la.unsqueeze(0), lb.unsqueeze(0), dx, dy)
    cov = _bidirectional_patch_coverage(ra_crop[0], cb_crop[0])
    ref_ov, curr_ov = _shift_align_pair(r_sub[-1], c_sub[0], dx, dy)
    static_mask = _compute_static_bg_mask(ref_ov, curr_ov)
    static_luma_err = luma_err
    if int(static_mask.sum().item()) >= 64:
        ys_r = (0.2126 * ref_ov[0] + 0.7152 * ref_ov[1] + 0.0722 * ref_ov[2])[static_mask]
        ys_c = (0.2126 * curr_ov[0] + 0.7152 * curr_ov[1] + 0.0722 * curr_ov[2])[static_mask]
        std_a = max(1e-4, float(ys_r.std().item()))
        std_b = max(1e-4, float(ys_c.std().item()))
        ns_r = (ys_r - ys_r.mean()) / ys_r.std().clamp(min=0.08)
        ns_c = (ys_c - ys_c.mean()) / ys_c.std().clamp(min=0.08)
        static_luma_err = float((ns_r - ns_c).abs().clamp(max=1.0).mean().item())
    lv_r = (r_sub[-1].clamp(0.0, 0.999) * 8.0).long()
    lv_c = (c_sub[0].clamp(0.0, 0.999) * 8.0).long()
    hr = torch.bincount(((lv_r[0] * 8 + lv_r[1]) * 8 + lv_r[2]).reshape(-1), minlength=512).float()
    hc = torch.bincount(((lv_c[0] * 8 + lv_c[1]) * 8 + lv_c[2]).reshape(-1), minlength=512).float()
    hist_l1 = 0.5 * float((hr / hr.sum().clamp(min=1.0) - hc / hc.sum().clamp(min=1.0)).abs().sum().item())
    if known_continuation:
        extreme_env_cut = bool(hist_l1 > 0.82 and luma_err > 0.68 and static_luma_err > 0.62 and cov < 0.015)
        same_scene_safe = not extreme_env_cut
    else:
        near_flat_same = max(std_a, std_b) < 0.025 and luma_err < 0.22
        same_scene_safe = bool(((luma_err < 0.38 or static_luma_err < 0.32) and (cov > 0.08)) or near_flat_same)

    step_l1 = float((c_sub[0] - r_sub[-1]).abs().mean().item())
    is_flash = False
    if int(c_sub.shape[0]) >= 4:
        peak_l1 = max(step_l1, float((c_sub[1] - r_sub[-1]).abs().mean().item()))
        pre_med = r_sub.median(dim=0).values
        post_med = c_sub[2:].median(dim=0).values
        sustained_l1 = float((post_med - pre_med).abs().mean().item())
        if peak_l1 > 0.04 and sustained_l1 < 0.38 * peak_l1:
            is_flash = True

    ref_lab = _srgb_to_lab(ref_ov.movedim(0, -1))[static_mask]
    curr_lab = _srgb_to_lab(curr_ov.movedim(0, -1))[static_mask]
    de_seam = float((ref_lab.mean(dim=0) - curr_lab.mean(dim=0)).norm().item()) if ref_lab.numel() > 0 else 0.0

    de_anchor = 0.0
    if scene_anchor is not None and isinstance(scene_anchor, torch.Tensor) and int(scene_anchor.shape[0]) > 0:
        a_sub = F.interpolate(scene_anchor[-min(4, int(scene_anchor.shape[0])):, ..., :3].to(device=dev).float().movedim(-1, 1), size=(h_s, 128), mode="area")
        mid_i = min(int(c_sub.shape[0]) - 1, max(0, int(c_sub.shape[0]) // 2))
        adx, ady, _ = _register_translation_128(_bt709_norm_luma(a_sub[-1:])[0], _bt709_norm_luma(c_sub[mid_i:mid_i + 1])[0])
        a_ov, c_mid_ov = _shift_align_pair(a_sub[-1], c_sub[mid_i], adx, ady)
        amask = _compute_static_bg_mask(a_ov, c_mid_ov)
        de_anchor = float((_srgb_to_lab(a_ov.movedim(0, -1))[amask].mean(dim=0) - _srgb_to_lab(c_mid_ov.movedim(0, -1))[amask].mean(dim=0)).norm().item())

    de_eff = max(de_seam, de_anchor)
    std_ratio = float(std_b / std_a)
    if (not same_scene_safe) or is_flash:
        alpha_auto = 0.0
        reason = "impact_flash_bypass" if is_flash else "hard_cut_bypass"
    elif de_eff < float(deadzone_de) and abs(std_ratio - 1.0) < 0.04:
        alpha_auto = 0.0
        reason = "deadzone_passthrough"
    else:
        if known_continuation:
            alpha_auto = max(0.35, min(1.0, (de_eff - float(deadzone_de)) / 4.5)) if de_eff >= float(deadzone_de) else min(1.0, abs(std_ratio - 1.0) * 5.0)
        else:
            alpha_auto = max(0.0, min(1.0, (de_eff - float(deadzone_de)) / 6.0 if de_eff >= float(deadzone_de) else abs(std_ratio - 1.0) * 5.0))
        reason = "drift_detected"

    out.update({
        "same_scene_safe": same_scene_safe, "is_flash": bool(is_flash),
        "luma_err": round(luma_err, 4), "coverage": round(cov, 4), "hist_l1": round(hist_l1, 4),
        "dx": int(dx), "dy": int(dy),
        "delta_e_seam": round(de_seam, 3), "delta_e_anchor": round(de_anchor, 3), "delta_e_eff": round(de_eff, 3),
        "luma_std_ratio": round(std_ratio, 3), "static_share": round(float(static_mask.float().mean().item()), 3),
        "alpha_auto": round(alpha_auto, 4), "reason": reason,
    })
    return out


def _clamped_mkl_lab_transform(
    src_pixels: torch.Tensor,
    ref_pixels: torch.Tensor,
    scale_bounds: tuple[float, float] = (0.85, 1.18),
    max_shift: tuple[float, float, float] = (18.0, 15.0, 15.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded MKL transfer clamping singular stretch and Lab channel shifts."""
    T, mu_s, mu_r = _mkl_transform_matrix(src_pixels, ref_pixels)
    u, s, vh = torch.linalg.svd(T)
    T_safe = u @ torch.diag(s.clamp(min=scale_bounds[0], max=scale_bounds[1])) @ vh
    lim = torch.tensor(max_shift, device=mu_s.device, dtype=mu_s.dtype).view(1, 3)
    mu_r_safe = mu_s + (mu_r - mu_s).clamp(min=-lim, max=lim)
    return T_safe, mu_s, mu_r_safe


def _apply_lab_transfer_chunked(
    frames: torch.Tensor,
    t_mat: torch.Tensor,
    mu_s: torch.Tensor,
    mu_r: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply bounded Lab MKL transfer in 32-frame chunks with dark-shadow protection."""
    n = int(frames.shape[0])
    out = frames.clone()
    for s0 in range(0, n, 32):
        s1 = min(n, s0 + 32)
        chunk = frames[s0:s1, ..., :3].float()
        lab = _srgb_to_lab(chunk)
        matched = ((lab.reshape(-1, 3) - mu_s) @ t_mat + mu_r).reshape_as(lab)
        shadow_gate = ((lab[..., :1] - 1.0) / 3.0).clamp(0.0, 1.0)
        w = weights[s0:s1].view(-1, 1, 1, 1) * shadow_gate
        out[s0:s1, ..., :3] = _lab_to_srgb(lab * (1.0 - w) + matched * w).to(dtype=frames.dtype)
    return out


def auto_dual_mkl_lab_match(
    images: torch.Tensor,
    reference_clean_tail: Optional[torch.Tensor],
    scene_anchor_clean_ref: Optional[torch.Tensor] = None,
    seam_frames: int = 14,
    deadzone_de: float = 1.5,
    global_lock_strength: float = 0.75,
    known_continuation: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    3-gate auto-color detector + two-layer anti-snowball bounded CIE-Lab MKL matching:
    Layer 1: Full-shot global drift lock (70% scene-start anchor + 30% prev-tail) on static pixels.
    Layer 2: Smooth quintic residual release across the first `seam_frames` frames.
    """
    gate = detect_seam_gate_and_drift(
        images,
        reference_clean_tail,
        scene_anchor=scene_anchor_clean_ref,
        deadzone_de=deadzone_de,
        known_continuation=known_continuation,
    )
    gate["applied_color"] = False
    gate["delta_e_after"] = float(gate.get("delta_e_eff", 0.0))
    if gate["alpha_auto"] <= 0.0 or reference_clean_tail is None or not isinstance(reference_clean_tail, torch.Tensor):
        return images, gate

    n_total, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    dev = images.device
    step = max(1, int(math.sqrt(h * w / 16384)))
    ref = reference_clean_tail[-min(4, int(reference_clean_tail.shape[0])):, ..., :3].to(device=dev, dtype=torch.float32)

    # Static pixel filter between reference[-1] and aligned seam head
    ref_small = F.interpolate(ref[-1:].movedim(-1, 1), size=(max(24, round(h * 128.0 / max(1, w))), 128), mode="area")
    cur_small = F.interpolate(images[:1, ..., :3].float().movedim(-1, 1), size=ref_small.shape[2:], mode="area")
    r_ov, c_ov = _shift_align_pair(ref_small[0], cur_small[0], gate["dx"], gate["dy"])
    smask_low = _compute_static_bg_mask(r_ov, c_ov)
    sub_src_img = images[:min(3, n_total), ::step, ::step, :3].float()
    sh_sub, sw_sub = int(sub_src_img.shape[1]), int(sub_src_img.shape[2])
    if int(ref.shape[1]) != h or int(ref.shape[2]) != w:
        sub_ref_img = F.interpolate(ref.movedim(-1, 1), size=(sh_sub, sw_sub), mode="area").movedim(1, -1)
    else:
        sub_ref_img = ref[:, ::step, ::step, :3]
    smask_full = F.interpolate(smask_low[None, None].float(), size=(sh_sub, sw_sub), mode="nearest")[0, 0] > 0.5
    if float(smask_full.float().mean().item()) < 0.15:
        smask_full = torch.ones_like(smask_full, dtype=torch.bool)

    src_lab_sub = _srgb_to_lab(sub_src_img)[:, smask_full].reshape(-1, 3)
    tail_lab_sub = _srgb_to_lab(sub_ref_img)[:, smask_full].reshape(-1, 3)

    out = images
    alpha_auto = float(gate["alpha_auto"])
    w_glob = alpha_auto * max(0.0, min(1.0, float(global_lock_strength)))
    if w_glob > 0.02:
        if scene_anchor_clean_ref is not None and isinstance(scene_anchor_clean_ref, torch.Tensor) and int(scene_anchor_clean_ref.shape[0]) > 0:
            anc_raw = scene_anchor_clean_ref[-min(4, int(scene_anchor_clean_ref.shape[0])):, ..., :3].to(device=dev, dtype=torch.float32)
            anc = F.interpolate(anc_raw.movedim(-1, 1), size=(sh_sub, sw_sub), mode="area").movedim(1, -1) if (int(anc_raw.shape[1]) != h or int(anc_raw.shape[2]) != w) else anc_raw[:, ::step, ::step, :]
            anc_lab = _srgb_to_lab(anc)[:, smask_full].reshape(-1, 3)
            n_a = min(int(anc_lab.shape[0]), 7000)
            n_t = max(1, int(round(n_a * 3.0 / 7.0)))
            target_global = torch.cat([anc_lab[:n_a], tail_lab_sub[:min(int(tail_lab_sub.shape[0]), n_t)]], dim=0)
        else:
            target_global = tail_lab_sub
        mid_probe_start = max(0, min(n_total - 3, n_total // 4))
        mid_lab_sub = _srgb_to_lab(images[mid_probe_start:mid_probe_start + 3, ::step, ::step, :3].float())[:, smask_full].reshape(-1, 3)
        tg, mug_s, mug_r = _clamped_mkl_lab_transform(torch.cat([src_lab_sub, mid_lab_sub], dim=0), target_global)
        glob_w = torch.full((n_total,), w_glob, device=dev, dtype=torch.float32)
        out = _apply_lab_transfer_chunked(out, tg, mug_s, mug_r, glob_w)

    n_seam = min(max(1, int(seam_frames)), n_total)
    head_lab_sub = _srgb_to_lab(out[:min(3, n_seam), ::step, ::step, :3].float())[:, smask_full].reshape(-1, 3)
    ts, mus_s, mus_r = _clamped_mkl_lab_transform(head_lab_sub, tail_lab_sub)
    u = torch.arange(n_seam, device=dev, dtype=torch.float32) / float(max(1, n_seam))
    quintic = (1.0 - (u ** 3) * (10.0 - 15.0 * u + 6.0 * (u ** 2))) * alpha_auto
    out = out.clone()
    out[:n_seam] = _apply_lab_transfer_chunked(out[:n_seam], ts, mus_s, mus_r, quintic)
    post_head_lab = _srgb_to_lab(out[:min(3, n_seam), ::step, ::step, :3].float())[:, smask_full].reshape(-1, 3)
    if int(post_head_lab.shape[0]) > 0 and int(tail_lab_sub.shape[0]) > 0:
        de_after = float(torch.norm(post_head_lab.mean(dim=0) - tail_lab_sub.mean(dim=0)).item())
        gate["delta_e_after"] = round(de_after, 3)
    gate["applied_color"] = True
    return out, gate


def heal_single_video_o1(
    images: torch.Tensor,
    audio: Optional[dict] = None,
    fps: int = 24,
    segment_seconds: float = 0.0,
    seam_frames: int = 14,
    bridge_max_frames: int = 4,
    deadzone_de: float = 1.5,
    global_lock_strength: float = 0.75,
    jump_spike_threshold: float = 1.65,
    search_radius: int = 11,
) -> tuple[torch.Tensor, Optional[dict], dict[str, Any]]:
    """O(1)-copy single-video seam healer: heals boundaries using slice views and 1 final torch.cat."""
    from .rife_micro_bridge import apply_zero_delta_rife_micro_bridge

    n = int(images.shape[0])
    rep: dict[str, Any] = {"mode": "single_video_auto_heal", "input_frames": n, "seams": []}
    if n < 12:
        return images, audio, rep

    fps_f = max(1.0, float(fps))
    sm_parts: list[torch.Tensor] = []
    for s0 in range(0, n, 128):
        s1 = min(n, s0 + 128)
        sm_parts.append(F.interpolate(images[s0:s1, ..., :3].float().movedim(-1, 1), size=(32, 32), mode="area"))
    sm_all = torch.cat(sm_parts, dim=0)
    l1 = (sm_all[1:] - sm_all[:-1]).abs().mean(dim=(1, 2, 3)) if sm_all.shape[0] > 1 else torch.zeros((0,), device=images.device)

    seams: list[int] = []
    min_sep = max(12, round(0.8 * fps_f))
    if float(segment_seconds) > 0.0:
        step_f = max(8, round(float(segment_seconds) * fps_f))
        rad = max(0, int(search_radius))
        for anc in range(step_f, n - 6, step_f):
            lo, hi = max(4, anc - rad), min(n - 4, anc + rad)
            if lo <= hi and l1.numel() >= hi:
                best = lo + int(torch.argmax(l1[lo - 1:hi]).item())
                if all(abs(best - s) >= 8 for s in seams):
                    seams.append(best)
    elif l1.numel() >= 11:
        med_l1 = float(l1.median().item())
        thresh = max(0.012, 1.8 * max(med_l1, 0.004))
        cands: list[tuple[float, int]] = []
        for t in range(6, n - 6):
            val = float(l1[t - 1].item())
            if val < thresh:
                continue
            pre_m = sm_all[max(0, t - 4):t].median(dim=0).values
            post_m = sm_all[t + 2:min(n, t + 6)].median(dim=0).values if t + 2 < n else sm_all[t]
            sustained = float((post_m - pre_m).abs().mean().item())
            if val > 0.04 and sustained < 0.35 * val:
                continue  # Ignore 1-frame impact flash
            cands.append((val, t))
        cands.sort(key=lambda x: x[0], reverse=True)
        for _, t in cands:
            if all(abs(t - s) >= min_sep for s in seams):
                seams.append(t)
        seams.sort()
    if not seams:
        return images, audio, rep

    bounds = [0] + sorted(seams) + [n]
    parts = [images[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    healed = [parts[0]]
    scene_anchor = parts[0][-4:]
    for i in range(1, len(parts)):
        prev_tail = healed[-1][-4:]
        seg, c_gate = auto_dual_mkl_lab_match(
            parts[i], prev_tail, scene_anchor_clean_ref=scene_anchor,
            seam_frames=seam_frames, deadzone_de=deadzone_de, global_lock_strength=global_lock_strength,
        )
        if not c_gate.get("same_scene_safe", False):
            scene_anchor = parts[i][-4:]
        seg, r_info = apply_zero_delta_rife_micro_bridge(
            seg, prev_tail, mode="auto_jump_rife", max_bridge_frames=bridge_max_frames,
            jump_spike_threshold=jump_spike_threshold, gate_info=c_gate,
        )
        healed.append(seg)
        rep["seams"].append({"frame": bounds[i], "color": c_gate, "rife": r_info})
    return torch.cat(healed, dim=0), audio, rep

