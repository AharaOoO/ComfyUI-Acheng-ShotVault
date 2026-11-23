import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .fuzzpuppy_engine import (
        analyze_seam_eI,
        build_subject_alignment_eS,
        run_rife_bridge,
        warp_composed_eT,
        _interp_alignment_flow,
        _quintic_release,
    )
except ImportError:
    import importlib.util as _ilu

    _fp_spec = _ilu.spec_from_file_location("acheng_fuzzpuppy_engine", Path(__file__).resolve().parent / "fuzzpuppy_engine.py")
    _fp_mod = _ilu.module_from_spec(_fp_spec)
    _fp_spec.loader.exec_module(_fp_mod)
    analyze_seam_eI = _fp_mod.analyze_seam_eI
    build_subject_alignment_eS = _fp_mod.build_subject_alignment_eS
    run_rife_bridge = _fp_mod.run_rife_bridge
    warp_composed_eT = _fp_mod.warp_composed_eT
    _interp_alignment_flow = _fp_mod._interp_alignment_flow
    _quintic_release = _fp_mod._quintic_release


def _pixels(frames):
    return frames.detach().to(device="cpu", dtype=torch.float32).numpy()


def _to_u8(frames):
    arr = _pixels(frames)
    return np.ascontiguousarray(np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def _probe(frames, cv):
    height, width = frames.shape[1:3]
    scale = min(1.0, 256 / max(height, width))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    return np.stack([cv.resize(frame, size, interpolation=cv.INTER_AREA) for frame in _pixels(frames)])


def _gray(frame, cv):
    return cv.cvtColor(frame, cv.COLOR_RGB2GRAY)


def _gray8(frame, cv):
    return np.round(_gray(frame, cv).clip(0, 1) * 255).astype(np.uint8)


def _structure(left, right, cv):
    left, right = _gray(left, cv), _gray(right, cv)
    left_std, right_std = float(left.std()), float(right.std())
    if max(left_std, right_std) < 0.015:
        return 0.0
    left = (left - left.mean()) / max(left_std, 0.04)
    right = (right - right.mean()) / max(right_std, 0.04)
    return float(np.abs(left - right).clip(0, 2).mean())


def _mae(left, right):
    return float(np.abs(left - right).mean())


def _similarity(left, right, cv):
    identity = np.eye(2, 3, dtype=np.float32)
    height, width = left.shape[:2]
    if min(height, width) < 24:
        return identity
    previous = _gray8(right, cv)
    target = _gray8(left, cv)
    keypoints = cv.goodFeaturesToTrack(previous, 200, 0.015, 5)
    if keypoints is None or len(keypoints) < 8:
        return identity
    tracked, valid, _ = cv.calcOpticalFlowPyrLK(previous, target, keypoints, None)
    if tracked is None:
        return identity
    back, reverse_valid, _ = cv.calcOpticalFlowPyrLK(target, previous, tracked, None)
    if back is None:
        return identity
    good = valid.ravel().astype(bool) & reverse_valid.ravel().astype(bool)
    good &= np.linalg.norm((keypoints - back).reshape(-1, 2), axis=1) < 1.5
    if int(good.sum()) < 8:
        return identity
    matrix, inliers = cv.estimateAffinePartial2D(
        keypoints[good], tracked[good], method=cv.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 8 or float(inliers.mean()) < 0.48:
        return identity
    scale = float(np.linalg.norm(matrix[:, 0]))
    angle = abs(float(np.arctan2(matrix[1, 0], matrix[0, 0])))
    center = np.array([width / 2, height / 2])
    center_move = np.linalg.norm(matrix[:, :2] @ center + matrix[:, 2] - center)
    if not (0.86 <= scale <= 1.16 and angle <= 0.16 and center_move <= 0.18 * max(height, width)):
        return identity
    warped = cv.warpAffine(right, matrix, (width, height), borderMode=cv.BORDER_REFLECT_101)
    if _structure(left, warped, cv) + 0.008 >= _structure(left, right, cv):
        return identity
    return matrix.astype(np.float32)


def _color_parameters(left, right):
    gain = (left.std((0, 1)) / np.maximum(right.std((0, 1)), 0.04)).clip(0.85, 1.18)
    offset = (left.mean((0, 1)) - gain * right.mean((0, 1))).clip(-0.12, 0.12)
    return gain, offset


def _bad_head_prefix(previous, current, limit, cv):
    """Detect genuinely blown-out white-flash or black-drop head frames without misfiring on scene tilts or VFX."""
    luma = _gray(current.reshape(-1, current.shape[2], 3), cv).reshape(current.shape[:3])
    prior = float(_gray(previous, cv).mean())
    ref_lo = min(max(1, limit), len(luma) - 1)
    ref_hi = min(len(luma), ref_lo + 4)
    local_post = float(np.median(luma[ref_lo:ref_hi].mean((1, 2)))) if ref_hi > ref_lo else prior
    hi_ref = max(prior, local_post)
    lo_ref = min(prior, local_post)

    means = luma[:limit + 1].mean((1, 2))
    clipped_hi = (luma[:limit + 1] > 0.94).mean((1, 2))
    clipped_lo = (luma[:limit + 1] < 0.03).mean((1, 2))
    bad_white = ((clipped_hi > 0.35) & (means > hi_ref + 0.09)) | (means > hi_ref + 0.16)
    bad_black = (clipped_lo > 0.65) & (means < lo_ref - 0.14)
    bad = bad_white | bad_black
    first_clean = np.flatnonzero(~bad)
    return int(first_clean[0]) if first_clean.size else limit + 1


def _adaptive_bridge_count(jump_ratio, limit, next_step_hot=False):
    if jump_ratio < 2.15 and not next_step_hot:
        want = 1
    elif jump_ratio < 3.10:
        want = 2
    elif jump_ratio < 4.20:
        want = 3
    else:
        want = 4
    return max(1, min(int(limit), want))


def _release_head(frames, matrix, probe_shape, gain, offset, count, cv):
    frames = _pixels(frames[:count])
    height, width = frames.shape[1:3]
    probe_h, probe_w = probe_shape
    scale = np.diag([width / probe_w, height / probe_h, 1.0])
    full = (scale @ np.vstack([matrix, [0, 0, 1]]) @ np.linalg.inv(scale))[:2]
    identity = np.eye(2, 3)
    result = []
    for index, frame in enumerate(frames):
        progress = index / count
        amount = 1 - progress ** 2 * (3 - 2 * progress)
        if not np.allclose(full, identity):
            frame = cv.warpAffine(
                frame, identity + amount * (full - identity), (width, height), borderMode=cv.BORDER_REFLECT_101
            )
        result.append((frame + amount * (frame * (gain - 1) + offset)).clip(0, 1))
    return np.stack(result)


def _fuzzpuppy_align_head(left_frames, right_frames, release_frames, fps=24.0):
    """Run FuzzPuppy eI + eS non-rigid dense LK + edge-aware diffusion + B-tracking on IMAGE tensors."""
    left_u8 = _to_u8(left_frames)
    right_u8 = _to_u8(right_frames)
    analysis = analyze_seam_eI(
        left_u8,
        right_u8,
        fps=fps,
        max_search_sec=0.0,
        overlap_frames=0,
        structure_threshold=0.38,
        coverage_threshold=0.08,
        preserve_duration=True,
    )
    if not analysis["safe"]:
        return None, analysis
    rel_sec = max(0.1, float(release_frames) / float(fps))
    subj = build_subject_alignment_eS(left_u8, right_u8, analysis, release_seconds=rel_sec)
    count = min(int(release_frames), len(right_u8))
    tr_full = analysis["transform"]
    color = analysis["color"]
    align_w = subj["width"]
    repaired = []
    for i in range(count):
        t_b = i / float(fps)
        amount = _quintic_release(t_b, rel_sec)
        flow_t = _interp_alignment_flow(subj, t_b)
        warped = warp_composed_eT(right_u8[i], tr_full, flow_t, align_w, amount, color=color)
        repaired.append(warped.astype(np.float32) / 255.0)
    analysis["eS_coverage"] = round(subj["coverage"], 5)
    analysis["eS_residual"] = subj["residual"]
    return np.stack(repaired, axis=0), analysis


def _flow_bridge(endpoints, count, cv):
    left, right = _pixels(endpoints)
    height, width = left.shape[:2]
    factor = min(1.0, 480 / max(height, width))
    size = max(16, round(width * factor)), max(16, round(height * factor))
    gray_left = cv.resize(_gray8(left, cv), size, interpolation=cv.INTER_AREA)
    gray_right = cv.resize(_gray8(right, cv), size, interpolation=cv.INTER_AREA)
    flow = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    scale = np.array([width / size[0], height / size[1]], dtype=np.float32)
    forward = cv.resize(flow.calc(gray_left, gray_right, None), (width, height)) * scale
    backward = cv.resize(flow.calc(gray_right, gray_left, None), (width, height)) * scale
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    grid = np.stack([grid_x, grid_y], axis=-1)
    result = []
    for index in range(1, count + 1):
        time = index / (count + 1)
        map_left = _inverse_displacement(grid, forward, time, cv)
        map_right = _inverse_displacement(grid, backward, 1 - time, cv)
        warped_left = cv.remap(left, map_left, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        warped_right = cv.remap(right, map_right, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        disagreement = np.abs(warped_left - warped_right).max(axis=2, keepdims=True)
        steep = np.clip((time - 0.30) / 0.40, 0, 1)
        weight = np.where(disagreement > 0.16, steep * steep * (3 - 2 * steep), time)
        result.append(warped_left * (1 - weight) + warped_right * weight)
    return np.stack(result).clip(0, 1)


def _inverse_displacement(grid, displacement, scale, cv):
    coordinates = grid
    for _iteration in range(2):
        sampled = cv.remap(displacement, coordinates, None, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        coordinates = grid - scale * sampled
    return coordinates


def _rife_bridge(endpoints, count, checkpoint):
    ep_u8 = _to_u8(endpoints)
    backend_choice = "rife422lite_onnx" if checkpoint == "rife422lite_fp32.onnx" else checkpoint
    frames_u8, tag = run_rife_bridge(ep_u8[0], ep_u8[1], count, backend_choice=backend_choice)
    return torch.from_numpy(frames_u8.astype(np.float32) / 255.0), tag


def _run_bridge_backend(endpoints, count, backend, allow_flow_fallback, rife_checkpoint, cv, report):
    selected = backend
    if selected == "rife":
        try:
            repaired, tag = _rife_bridge(endpoints, count, rife_checkpoint)
            report["backend"] = tag
            return repaired, True
        except Exception as error:
            report["backend_error"] = str(error)
            if not allow_flow_fallback:
                return None, False
            selected = "flow"
    if selected == "flow":
        try:
            repaired = _flow_bridge(endpoints, count, cv)
            report["backend"] = "opencv_dis"
            return repaired, True
        except (cv.error, AttributeError, ValueError) as error:
            report["flow_error"] = str(error)
            return None, False
    return None, False


class AchengSeamBridge:
    CATEGORY = "Acheng/Video"
    FUNCTION = "repair"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio_unchanged", "diagnostics")
    DESCRIPTION = (
        "Repair one known IMAGE join with FuzzPuppy eI dual-gate hard-cut protection, "
        "eS dense LK edge-aware non-rigid alignment, or rife422lite ONNX bridge. No timing change."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Current segment after overlap removal, or a joined image batch."}),
                "mode": (["auto", "fuzzpuppy_align", "align_color", "flow", "rife", "off"], {"default": "auto"}),
                "join_frame": ("INT", {"default": 0, "min": 0, "max": 1000000,
                                       "tooltip": "0 = use previous_tail. Positive = index of first B frame in images (zero-based)."}),
                "max_bridge_frames": ("INT", {"default": 5, "min": 1, "max": 12}),
                "release_frames": ("INT", {"default": 19, "min": 1, "max": 48}),
                "sensitivity_ratio": ("FLOAT", {"default": 1.65, "min": 1.0, "max": 5.0, "step": 0.05}),
                "protect_hard_cuts": ("BOOLEAN", {"default": True}),
                "repair_white_head": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_tail": ("IMAGE", {"tooltip": "Last few delivered frames of the previous segment. Leave empty on the first segment."}),
                "audio": ("AUDIO", {"tooltip": "Optional passthrough. You can also keep audio wired directly to the encoder."}),
                "is_continuation": ("BOOLEAN", {"default": True, "tooltip": "Wire the loop continuation flag here to skip the first segment and intentional cuts."}),
                "auto_bridge_backend": (["flow", "rife"], {"default": "rife"}),
                "rife_checkpoint": (["rife422lite_fp32.onnx", "rife49.pth", "rife47.pth", "rife417.pth", "rife426.pth"], {"default": "rife422lite_fp32.onnx"}),
                "auto_align_color": ("BOOLEAN", {"default": True}),
            },
        }

    def repair(
        self,
        images,
        mode="auto",
        join_frame=0,
        max_bridge_frames=5,
        release_frames=19,
        sensitivity_ratio=1.65,
        protect_hard_cuts=True,
        repair_white_head=True,
        previous_tail=None,
        audio=None,
        is_continuation=True,
        auto_bridge_backend="rife",
        rife_checkpoint="rife422lite_fp32.onnx",
        auto_align_color=True,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
            raise ValueError("Acheng Seam Bridge expects non-empty RGB IMAGE input [frames, height, width, 3].")
        report = {
            "mode": mode,
            "join_frame": int(join_frame),
            "output_frames": len(images),
            "frame_delta": 0,
            "changed_frames": 0,
            "backend": "none",
        }

        def finish(reason, output=images):
            return output, audio, json.dumps(dict(report, reason=reason), ensure_ascii=False)

        if mode == "off" or not is_continuation:
            return finish("disabled_or_not_continuation")
        seam = int(join_frame)
        if seam < 0 or seam >= len(images):
            raise ValueError("join_frame must be zero (external tail) or index an existing B frame.")
        left = images[max(0, seam - 8):seam] if seam else previous_tail
        right = images[seam:]
        if left is None:
            return finish("no_previous_tail")
        if not isinstance(left, torch.Tensor) or left.ndim != 4 or left.shape[1:] != right.shape[1:]:
            return finish("incompatible_frame_geometry")
        if not len(left) or len(right) < 2:
            return finish("not_enough_frames")
        try:
            import cv2 as cv
        except ImportError:
            return finish("opencv_unavailable")

        limit = max(1, min(int(max_bridge_frames), len(right) - 1))

        # Dedicated FuzzPuppy eS non-rigid dense LK + edge diffusion + 0.8s quintic release mode
        if mode == "fuzzpuppy_align":
            repaired_np, analysis = _fuzzpuppy_align_head(left[-8:], right, release_frames=release_frames, fps=24.0)
            report.update(
                structureError=analysis["structureError"],
                flowCoverage=analysis["flowCoverage"],
                safe=analysis["safe"],
            )
            if repaired_np is None:
                return finish("hard_cut_or_weak_correspondence")
            repaired = torch.from_numpy(repaired_np).to(images)
            result = images.clone()
            cnt = len(repaired)
            result[seam:seam + cnt] = repaired
            report["backend"] = "fuzzpuppy_eS_dense_lk"
            report["eS_coverage"] = analysis.get("eS_coverage")
            report["eS_residual"] = analysis.get("eS_residual")
            report["changed_frames"] = cnt
            return finish("repaired", result)

        # Unconditional direct bridge when protect_hard_cuts is explicitly False
        if mode in ("rife", "flow") and not protect_hard_cuts:
            count = limit
            endpoints = torch.stack((left[-1].to(right), right[count]))
            repaired, ok = _run_bridge_backend(
                endpoints, count, mode, allow_flow_fallback=(mode == "rife"), rife_checkpoint=rife_checkpoint, cv=cv, report=report
            )
            if not ok or repaired is None:
                return finish("bridge_backend_failed")
            repaired = torch.as_tensor(repaired).to(images)
            if not bool(torch.isfinite(repaired).all()):
                return finish("nonfinite_repair")
            result = images.clone()
            result[seam:seam + count] = repaired
            report["mutation_kind"] = "unconditional_bridge"
            report["repair_offset"] = 0
            report["b_endpoint_frame"] = seam + count
            report["changed_frames"] = count
            return finish("repaired", result)

        # FuzzPuppy eI exact dual-gate hard-cut protection (structureError < 0.38 AND flowCoverage > 0.08)
        if protect_hard_cuts:
            ei = analyze_seam_eI(
                _to_u8(left[-4:]),
                _to_u8(right[:min(len(right), 8)]),
                fps=24.0,
                max_search_sec=0.0,
                overlap_frames=0,
                structure_threshold=0.38,
                coverage_threshold=0.08,
                preserve_duration=True,
            )
            report.update(
                structure_error=ei["structureError"],
                flow_coverage=ei["flowCoverage"],
                ei_safe=ei["safe"],
            )
            if not ei["safe"]:
                return finish("hard_cut_or_weak_correspondence")

        probe_len = min(len(right), limit + 12)
        left_probe = _probe(left[-8:], cv)
        right_probe = _probe(right[:probe_len], cv)
        if not np.isfinite(left_probe).all() or not np.isfinite(right_probe).all():
            return finish("nonfinite_probe")

        anchor = left_probe[-1]
        white = _bad_head_prefix(anchor, right_probe, limit, cv) if repair_white_head else 0
        report["white_head_frames"] = white
        if white > limit:
            return finish("no_clean_endpoint_in_bridge_budget")

        score_index = white if white else 0
        candidate = right_probe[score_index]
        matrix = _similarity(anchor, candidate, cv)
        probe_h, probe_w = candidate.shape[:2]
        aligned = cv.warpAffine(candidate, matrix, (probe_w, probe_h), borderMode=cv.BORDER_REFLECT_101)
        d0 = _mae(anchor, right_probe[0])
        gain, offset = _color_parameters(anchor, aligned)
        aligned_rgb = (aligned * gain + offset).clip(0, 1)
        corrected = _mae(anchor, aligned_rgb)
        report["join_mae_before"] = round(d0, 5)

        steady_tail_start = min(max(score_index + 3, 4), max(0, len(right_probe) - 3))
        steady_parts = []
        if len(left_probe) >= 2:
            steady_parts.append(np.abs(np.diff(left_probe, axis=0)).mean((1, 2, 3)))
        if len(right_probe) - steady_tail_start >= 2:
            steady_parts.append(np.abs(np.diff(right_probe[steady_tail_start:], axis=0)).mean((1, 2, 3)))
        elif len(right_probe[score_index:]) >= 2:
            steady_parts.append(np.abs(np.diff(right_probe[score_index:], axis=0)).mean((1, 2, 3)))
        diffs = np.concatenate(steady_parts) if steady_parts else np.array([0.007], dtype=np.float32)
        typical = max(0.007, float(np.median(diffs))) if diffs.size else 0.007

        head_steps = [d0] + [_mae(right_probe[k - 1], right_probe[k]) for k in range(1, min(6, len(right_probe)))]
        sens = float(sensitivity_ratio)
        direct_jump = d0 >= max(0.014, typical * sens)

        mutation_kind = "none"
        rel_offset = 0
        bridge_count = limit
        trigger_mae = d0
        if white > 0:
            mutation_kind = "bad_head_flash"
            bridge_count = min(limit, max(white, 1))
            trigger_mae = max(d0, head_steps[min(white, len(head_steps) - 1)])
        elif direct_jump:
            mutation_kind = "direct_seam_jump"
            next_hot = len(head_steps) > 1 and head_steps[1] > typical * 1.45
            bridge_count = _adaptive_bridge_count(d0 / typical, limit, next_step_hot=next_hot)
        else:
            max_k = min(4, len(head_steps) - 1, len(right) - 3)
            for k in range(1, max_k + 1):
                dk = head_steps[k]
                prev_max = max(head_steps[:k])
                if typical >= 0.009 and prev_max <= typical * 0.58 and dk >= max(0.016, typical * max(1.45, sens * 0.90)):
                    if k + 1 <= len(right) - 1:
                        mutation_kind = "head_freeze_snap"
                        bridge_count = min(limit, max(k + 1, 2))
                        trigger_mae = dk
                        break
                next_calm = (k + 1 >= len(head_steps)) or (head_steps[k + 1] < dk * 0.78)
                if dk >= max(0.020, typical * max(1.75, sens * 1.05)) and next_calm:
                    span = _adaptive_bridge_count(dk / typical, min(limit, 3))
                    pre_f = min(k, max(1, (span + 1) // 2))
                    cand_off = max(0, k - pre_f)
                    cand_cnt = min(span, len(right) - 1 - cand_off)
                    if cand_off + cand_cnt >= k and cand_cnt >= 1:
                        mutation_kind = "delayed_token_jump"
                        rel_offset = cand_off
                        bridge_count = cand_cnt
                        trigger_mae = dk
                        break

        report["jump_ratio"] = round(trigger_mae / typical, 3)
        report["mutation_kind"] = mutation_kind
        can_align = d0 > 0.016 and corrected < d0 * 0.78 and not white

        selected = mode
        if selected == "auto":
            if white > 0:
                selected = auto_bridge_backend
            elif auto_align_color and can_align and (direct_jump or corrected < 0.025):
                selected = "fuzzpuppy_align"
            elif mutation_kind != "none":
                selected = auto_bridge_backend
            else:
                return finish("smooth_passthrough")
        elif selected in ("rife", "flow"):
            if mutation_kind == "none":
                mutation_kind = "unconditional_bridge"
                report["mutation_kind"] = mutation_kind
            rel_offset = 0
            bridge_count = min(limit, max(white, limit))

        if selected == "fuzzpuppy_align":
            repaired_np, analysis = _fuzzpuppy_align_head(left[-8:], right, release_frames=release_frames, fps=24.0)
            if repaired_np is None:
                return finish("alignment_not_helpful_or_white_head")
            repaired = torch.from_numpy(repaired_np).to(images)
            rep_start = seam
            count = len(repaired)
            report["backend"] = "fuzzpuppy_eS_dense_lk"
            left_ep_probe = anchor
            if count < len(right):
                right_ep_probe = _probe(right[count:count + 1], cv)[0]
                raw_mid = right_probe[:count] if count <= len(right_probe) else _probe(right[:count], cv)
                raw_window_probe = np.concatenate([anchor[None], raw_mid, right_ep_probe[None]], axis=0)
            else:
                right_ep_probe = None
                raw_mid = right_probe[:count] if count <= len(right_probe) else _probe(right[:count], cv)
                raw_window_probe = np.concatenate([anchor[None], raw_mid], axis=0)
        elif selected == "align_color":
            if not can_align:
                return finish("alignment_not_helpful_or_white_head")
            count = min(int(release_frames), len(right) - 1)
            repaired = _release_head(right, matrix, (probe_h, probe_w), gain, offset, count, cv)
            rep_start = seam
            report["backend"] = "opencv_similarity_rgb"
            left_ep_probe = anchor
            right_ep_probe = _probe(right[count:count + 1], cv)[0]
            raw_window_probe = np.concatenate([anchor[None], right_probe[:count + 1]], axis=0)
        else:
            count = max(1, min(int(bridge_count), len(right) - 1 - rel_offset))
            left_ep_frame = left[-1].to(right) if rel_offset == 0 else right[rel_offset - 1]
            right_ep_idx = rel_offset + count
            right_ep_frame = right[right_ep_idx]
            left_ep_probe = anchor if rel_offset == 0 else right_probe[rel_offset - 1]
            right_ep_probe = right_probe[right_ep_idx] if right_ep_idx < len(right_probe) else _probe(right_ep_frame[None], cv)[0]

            endpoints = torch.stack((left_ep_frame, right_ep_frame))
            repaired, ok = _run_bridge_backend(
                endpoints, count, selected, allow_flow_fallback=(selected == "rife"), rife_checkpoint=rife_checkpoint, cv=cv, report=report
            )
            if not ok or repaired is None:
                return finish("bridge_backend_failed")
            rep_start = seam + rel_offset
            report["repair_offset"] = rel_offset
            report["b_endpoint_frame"] = seam + right_ep_idx
            raw_mid = right_probe[rel_offset:right_ep_idx] if right_ep_idx <= len(right_probe) else _probe(right[rel_offset:right_ep_idx], cv)
            raw_window_probe = np.concatenate([left_ep_probe[None], raw_mid, right_ep_probe[None]], axis=0)

        repaired = torch.as_tensor(repaired).to(images)
        if not bool(torch.isfinite(repaired).all()):
            return finish("nonfinite_repair")
        repaired_probe = _probe(repaired, cv)
        after_first = _mae(left_ep_probe, repaired_probe[0])
        report["join_mae_after"] = round(after_first, 5)

        if mode == "auto":
            raw_peak = float(np.abs(np.diff(raw_window_probe, axis=0)).mean((1, 2, 3)).max())
            edge_parts = [left_ep_probe[None], repaired_probe]
            if right_ep_probe is not None:
                edge_parts.append(right_ep_probe[None])
            edges = np.concatenate(edge_parts, axis=0)
            peak = float(np.abs(np.diff(edges, axis=0)).mean((1, 2, 3)).max())
            report["raw_peak_mae"] = round(raw_peak, 5)
            report["repaired_peak_mae"] = round(peak, 5)
            if white == 0 and peak >= raw_peak * 0.94:
                return finish("repair_did_not_reduce_peak_jump")
            if peak > max(raw_peak * 1.05, typical * 2.4):
                return finish("repair_introduced_a_larger_jump")

        result = images.clone()
        result[rep_start:rep_start + count] = repaired
        report["changed_frames"] = count
        return finish("repaired", result)
