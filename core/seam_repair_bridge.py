"""On-demand single-seam FuzzPuppy repair bridge for ShotVault Director Console & Assembler.

Reuses `ComfyUI-Acheng-SeamBridge/fuzzpuppy_engine.py` directly as the single source of truth
so there is zero duplicated algorithm code between #188 and the Director Console.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Optional

import av
import numpy as np
import torch

from ..compat import ensure_inside_root, resample_audio
from .grid_math import DELIVERY_AUDIO_SR, FPS_DEFAULT
from .seam_qc import write_mp4_with_pcm
from .vault_store import VaultStore

_FP_ENGINE_MOD = None


def _load_fuzzpuppy_engine():
    global _FP_ENGINE_MOD
    if _FP_ENGINE_MOD is not None:
        return _FP_ENGINE_MOD
    this_dir = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.dirname(os.path.dirname(this_dir))
    seam_pkg = os.path.join(custom_nodes_dir, "ComfyUI-Acheng-SeamBridge")
    engine_py = os.path.join(seam_pkg, "fuzzpuppy_engine.py")
    if not os.path.isfile(engine_py):
        raise FileNotFoundError(f"ComfyUI-Acheng-SeamBridge engine not found at {engine_py}")
    spec = importlib.util.spec_from_file_location("acheng_seambridge_fuzzpuppy_engine", engine_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fuzzpuppy_engine from {engine_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["acheng_seambridge_fuzzpuppy_engine"] = mod
    spec.loader.exec_module(mod)
    _FP_ENGINE_MOD = mod
    return mod


def _decode_mp4_frames_and_audio(
    mp4_path: str,
    from_end_frames: Optional[int] = None,
    from_start_frames: Optional[int] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    """Decode RGB uint8 frames + 48kHz stereo float32 audio (full clip or sliced tail/head)."""
    frames: list[np.ndarray] = []
    audio_chunks: list[np.ndarray] = []
    fps = float(FPS_DEFAULT)
    src_sr = DELIVERY_AUDIO_SR

    with av.open(mp4_path) as container:
        v_stream = next((s for s in container.streams if s.type == "video"), None)
        if v_stream is None:
            raise ValueError(f"No video stream in {mp4_path}")
        if v_stream.average_rate:
            fps = float(v_stream.average_rate)
        elif v_stream.base_rate:
            fps = float(v_stream.base_rate)

        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if from_start_frames is not None and len(frames) >= from_start_frames:
                break

        a_stream = next((s for s in container.streams if s.type == "audio"), None)
        if a_stream is not None:
            src_sr = int(a_stream.sample_rate or DELIVERY_AUDIO_SR)
            container.seek(0)
            for aframe in container.decode(audio=0):
                arr = aframe.to_ndarray()
                nb_ch = int(getattr(aframe.layout, "nb_channels", arr.shape[0] if arr.ndim > 1 else 1) or 1)
                if not aframe.format.is_planar and nb_ch > 1:
                    arr = arr.reshape(-1, nb_ch).T
                elif arr.ndim == 1:
                    arr = arr[None, :]
                arr_f = np.ascontiguousarray(arr, dtype=np.float32)
                if np.abs(arr_f).max(initial=0.0) > 1.5:
                    arr_f = arr_f / 32768.0
                if arr_f.shape[0] == 1:
                    arr_f = np.repeat(arr_f, 2, axis=0)
                elif arr_f.shape[0] > 2:
                    arr_f = arr_f[:2]
                audio_chunks.append(arr_f)

    if not frames:
        raise ValueError(f"Decoded 0 video frames from {mp4_path}")

    frames_arr = np.stack(frames, axis=0)
    total_f = len(frames_arr)

    audio_np: Optional[np.ndarray] = None
    if audio_chunks:
        raw_t = torch.from_numpy(np.concatenate(audio_chunks, axis=1))
        aud_48k = resample_audio(raw_t, src_sr, DELIVERY_AUDIO_SR).numpy()
        audio_np = np.ascontiguousarray(aud_48k, dtype=np.float32)

    if from_end_frames is not None and total_f > from_end_frames:
        keep_f = int(from_end_frames)
        frames_arr = frames_arr[-keep_f:]
        if audio_np is not None:
            keep_s = max(1, int(round(keep_f / fps * DELIVERY_AUDIO_SR)))
            audio_np = audio_np[:, -keep_s:] if audio_np.shape[1] >= keep_s else audio_np
    elif from_start_frames is not None:
        keep_f = min(total_f, int(from_start_frames))
        frames_arr = frames_arr[:keep_f]
        if audio_np is not None:
            keep_s = max(1, int(round(keep_f / fps * DELIVERY_AUDIO_SR)))
            audio_np = audio_np[:, :keep_s]

    return frames_arr, audio_np, fps


def _resolve_prev_and_curr_paths(
    store: VaultStore,
    shot_index: int,
    take_id: Optional[str] = None,
) -> tuple[str, str, str, int]:
    """Return `(prev_mp4_abs, curr_mp4_abs, curr_take_id, fps)` for repairing the seam before `shot_index`."""
    proj = store.load_project()
    fps = int(proj.get("fps") or FPS_DEFAULT)
    shots = proj.get("shots") or {}
    curr_rec = shots.get(str(int(shot_index))) or {}
    curr_tid = str(take_id or curr_rec.get("active_take_id") or "")
    if not curr_tid:
        raise ValueError(f"Shot #{shot_index} has no active take to repair.")
    curr_meta = (curr_rec.get("takes") or {}).get(curr_tid) or {}
    curr_rel = str(curr_meta.get("video_rel_path") or f"shots/{curr_tid}.mp4")
    curr_abs = ensure_inside_root(store.project_root, curr_rel)
    if not os.path.isfile(curr_abs):
        raise FileNotFoundError(f"Current take video not found: {curr_abs}")

    # Prefer previous shot's active take (matching ShotVaultAssemble) before falling back to historical parent_take_id / seed
    prev_abs = ""
    if int(shot_index) > 1:
        prev_rec = shots.get(str(int(shot_index) - 1)) or {}
        p_act = str(prev_rec.get("active_take_id") or "")
        if p_act:
            pm = (prev_rec.get("takes") or {}).get(p_act) or {}
            cand = ensure_inside_root(store.project_root, str(pm.get("video_rel_path") or f"shots/{p_act}.mp4"))
            if os.path.isfile(cand):
                prev_abs = cand

    prev_tid = str(curr_meta.get("parent_take_id") or "")
    if not prev_abs and prev_tid:
        for srec in shots.values():
            pm = (srec.get("takes") or {}).get(prev_tid)
            if pm:
                cand = ensure_inside_root(store.project_root, str(pm.get("video_rel_path") or f"shots/{prev_tid}.mp4"))
                if os.path.isfile(cand):
                    prev_abs = cand
                    break
        if not prev_abs:
            seed = proj.get("seed_shot") or {}
            if seed.get("take_id") == prev_tid:
                cand = ensure_inside_root(store.project_root, str(seed.get("video_rel_path") or "shots/shot_0000_seed.mp4"))
                if os.path.isfile(cand):
                    prev_abs = cand

    if not prev_abs and int(shot_index) == 1:
        seed = proj.get("seed_shot") or {}
        if seed.get("take_id"):
            cand = ensure_inside_root(store.project_root, str(seed.get("video_rel_path") or "shots/shot_0000_seed.mp4"))
            if os.path.isfile(cand):
                prev_abs = cand

    if not prev_abs:
        raise ValueError(f"Shot #{shot_index} has no preceding clip to build a seam repair.")
    return prev_abs, curr_abs, curr_tid, fps


def _align_frames_u8_to_target(frames_src: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize `[N, H, W, 3]` uint8 frames to `(target_h, target_w)` when adjacent takes differ in resolution."""
    if frames_src.ndim != 4 or (int(frames_src.shape[1]) == int(target_h) and int(frames_src.shape[2]) == int(target_w)):
        return frames_src
    try:
        import cv2  # type: ignore

        interp = cv2.INTER_AREA if (frames_src.shape[1] > target_h and frames_src.shape[2] > target_w) else cv2.INTER_CUBIC
        resized = [cv2.resize(fr, (int(target_w), int(target_h)), interpolation=interp) for fr in frames_src]
        return np.ascontiguousarray(np.stack(resized, axis=0), dtype=np.uint8)
    except Exception:
        t = torch.from_numpy(frames_src).permute(0, 3, 1, 2).float()
        out = torch.nn.functional.interpolate(t, size=(int(target_h), int(target_w)), mode="bicubic", align_corners=False)
        return np.ascontiguousarray(out.clamp(0.0, 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy())


def repair_shot_seam_preview(
    store: VaultStore,
    shot_index: int,
    take_id: Optional[str] = None,
    mode: str = "rife",
    rife_bridge_frames: int = 5,
    rife_backend: str = "auto",
    preserve_duration: bool = False,
    max_search_sec: float = 0.5,
    structure_threshold: float = 0.38,
    coverage_threshold: float = 0.08,
    release_seconds: float = 0.8,
    retry_policy: str = "off",
) -> dict[str, Any]:
    """Run FuzzPuppy repair on just the ~3.5s seam window around `prev_take -> curr_take` (takes ~2-4s, no H3 rerun)."""
    fp_eng = _load_fuzzpuppy_engine()
    prev_abs, curr_abs, curr_tid, fps_proj = _resolve_prev_and_curr_paths(store, shot_index, take_id=take_id)

    n_tail = max(24, int(round(1.5 * fps_proj)))
    n_head = max(36, int(round(2.0 * fps_proj)))
    frames_a, aud_a, fps_a = _decode_mp4_frames_and_audio(prev_abs, from_end_frames=n_tail)
    frames_b, aud_b, _ = _decode_mp4_frames_and_audio(curr_abs, from_start_frames=n_head)
    if frames_a.shape[1:3] != frames_b.shape[1:3]:
        frames_a = _align_frames_u8_to_target(frames_a, int(frames_b.shape[1]), int(frames_b.shape[2]))
    fps_val = fps_a if fps_a >= 12.0 else float(fps_proj)

    # Compute baseline original QC on un-repaired seam for honest side-by-side comparison
    raw_concat = np.concatenate([frames_a, frames_b], axis=0)
    raw_qc = fp_eng.evaluate_seam_window_qc(raw_concat, len(frames_a), seam_span=4, fps=fps_val)

    out_frames_u8, out_audio_np, report = fp_eng.merge_clips_fuzzpuppy(
        frames_a_u8=frames_a,
        frames_b_u8=frames_b,
        fps=fps_val,
        mode=mode,
        rife_bridge_frames=int(rife_bridge_frames),
        rife_backend=rife_backend,
        preserve_duration=bool(preserve_duration),
        max_search_sec=float(max_search_sec),
        overlap_frames=4,
        structure_threshold=float(structure_threshold),
        coverage_threshold=float(coverage_threshold),
        release_seconds=float(release_seconds),
        audio_a=aud_a,
        audio_b=aud_b,
        sample_rate=DELIVERY_AUDIO_SR,
        retry_policy=retry_policy,
    )

    repaired_rel = f"_vault/seam_repaired_{os.path.basename(curr_tid)}.mp4"
    repaired_abs = ensure_inside_root(store.project_root, repaired_rel)
    frames_t = torch.from_numpy(out_frames_u8).float() / 255.0
    aud_t = torch.from_numpy(out_audio_np).float() if out_audio_np is not None else None
    write_mp4_with_pcm(frames_t, aud_t, DELIVERY_AUDIO_SR, max(1, int(round(fps_val))), repaired_abs, crf=19)

    report["raw_seam_qc"] = raw_qc
    repair_cfg = {
        "enabled": True,
        "choice": "repaired" if report.get("safe") and report.get("applied") not in ("hard_cut_fallback", "lossless_hard_cut") else "original",
        "take_id": curr_tid,
        "mode": mode,
        "rife_bridge_frames": int(rife_bridge_frames),
        "rife_backend": rife_backend,
        "preserve_duration": bool(preserve_duration),
        "max_search_sec": float(max_search_sec),
        "structure_threshold": float(structure_threshold),
        "coverage_threshold": float(coverage_threshold),
        "release_seconds": float(release_seconds),
        "retry_policy": retry_policy,
        "repaired_seam_rel": repaired_rel,
        "report": report,
    }
    store.set_shot_seam_repair(shot_index, repair_cfg)
    return {"ok": True, "shot_index": int(shot_index), "take_id": curr_tid, "seam_repair": repair_cfg}


def apply_repaired_seam_to_clip_pair(
    clip_a_path: str,
    clip_b_path: str,
    repair_cfg: dict[str, Any],
    out_merged_path: str,
    fps: int = FPS_DEFAULT,
    codec: str = "h264",
) -> tuple[int, dict[str, Any]]:
    """Apply user-approved FuzzPuppy repair to full `(clip_a, clip_b)` pair for ShotVaultAssemble."""
    fp_eng = _load_fuzzpuppy_engine()
    frames_a, aud_a, fps_a = _decode_mp4_frames_and_audio(clip_a_path)
    frames_b, aud_b, _ = _decode_mp4_frames_and_audio(clip_b_path)
    if frames_a.shape[1:3] != frames_b.shape[1:3]:
        frames_a = _align_frames_u8_to_target(frames_a, int(frames_b.shape[1]), int(frames_b.shape[2]))
    fps_val = fps_a if fps_a >= 12.0 else float(fps)

    out_frames_u8, out_audio_np, report = fp_eng.merge_clips_fuzzpuppy(
        frames_a_u8=frames_a,
        frames_b_u8=frames_b,
        fps=fps_val,
        mode=str(repair_cfg.get("mode") or "rife"),
        rife_bridge_frames=int(repair_cfg.get("rife_bridge_frames") or 5),
        rife_backend=str(repair_cfg.get("rife_backend") or "auto"),
        preserve_duration=bool(repair_cfg.get("preserve_duration", False)),
        max_search_sec=float(repair_cfg.get("max_search_sec", 0.5)),
        overlap_frames=4,
        structure_threshold=float(repair_cfg.get("structure_threshold", 0.38)),
        coverage_threshold=float(repair_cfg.get("coverage_threshold", 0.08)),
        release_seconds=float(repair_cfg.get("release_seconds", 0.8)),
        audio_a=aud_a,
        audio_b=aud_b,
        sample_rate=DELIVERY_AUDIO_SR,
        retry_policy=str(repair_cfg.get("retry_policy") or "off"),
    )
    frames_t = torch.from_numpy(out_frames_u8).float() / 255.0
    aud_t = torch.from_numpy(out_audio_np).float() if out_audio_np is not None else None
    write_mp4_with_pcm(frames_t, aud_t, DELIVERY_AUDIO_SR, max(1, int(round(fps_val))), out_merged_path, crf=16, codec=codec)
    return len(out_frames_u8), report
