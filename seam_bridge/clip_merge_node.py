"""AchengFuzzPuppyClipMerge: ComfyUI VIDEO-to-VIDEO drop-in 1:1 port of FuzzPuppy Clip Merge.

Replaces raw ConcatenateVideo at segment joins so both Clip A's tail and Clip B's head
(video + audio) are available for FuzzPuppy's exact `eI` dual-gate hard-cut protection,
`eS` multi-timestamp dense LK + edge-aware diffusion + B-clip tracking + 0.8s quintic
release, `eT/V` symmetric 3/5/7-frame rife422lite ONNX bridge, and `eP` 30ms audio crossfade.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from fractions import Fraction
from typing import Any, Optional

import numpy as np
import torch

_SYNC_PROJ_LOCK = threading.Lock()


def _read_shot_repair_override(shot_pipe: Any) -> tuple[Optional[str], Optional[int], str, Optional[dict[str, Any]]]:
    """Read `(project_root, shot_index, take_id, seam_repair_cfg)` from ShotVault `project.json` if `shot_pipe` is wired."""
    if not isinstance(shot_pipe, dict):
        return None, None, "", None
    proj_root = str(shot_pipe.get("project_root") or "").strip()
    shot_idx = shot_pipe.get("current_shot_index")
    take_id = str(shot_pipe.get("last_saved_take_id") or "").strip()
    if not proj_root or shot_idx is None:
        return proj_root or None, int(shot_idx) if shot_idx is not None else None, take_id, None
    pjson = os.path.join(proj_root, "project.json")
    if not os.path.isfile(pjson):
        return proj_root, int(shot_idx), take_id, None
    try:
        with open(pjson, "r", encoding="utf-8") as f:
            proj = json.load(f)
        rec = (proj.get("shots") or {}).get(str(int(shot_idx))) or {}
        if not take_id:
            take_id = str(rec.get("active_take_id") or "")
        sr = rec.get("seam_repair")
        return proj_root, int(shot_idx), take_id, (dict(sr) if isinstance(sr, dict) and sr else None)
    except Exception:
        return proj_root, int(shot_idx), take_id, None


def _sync_repair_preview_to_vault(
    proj_root: Optional[str],
    shot_idx: Optional[int],
    take_id: str,
    out_frames_u8: np.ndarray,
    out_audio_np: Optional[np.ndarray],
    fps_val: float,
    sample_rate: int,
    seam_center_frame: int,
    repair_meta: dict[str, Any],
) -> None:
    """Persist a 3s repaired seam preview + report into ShotVault `project.json` so Director Console matches merged video 1:1."""
    if not proj_root or shot_idx is None or not take_id:
        return
    pjson = os.path.join(proj_root, "project.json")
    if not os.path.isfile(pjson):
        return
    try:
        import av

        safe_tid = os.path.basename(take_id)
        vault_dir = os.path.join(proj_root, "_vault")
        os.makedirs(vault_dir, exist_ok=True)
        repaired_rel = f"_vault/seam_repaired_{safe_tid}.mp4"
        repaired_abs = os.path.join(proj_root, repaired_rel)

        # Slice ~3s window around seam (1s before + 2s after) matching ShotVault's 3s original preview
        fps_i = max(1, int(round(fps_val)))
        s_idx = max(0, int(seam_center_frame) - fps_i)
        e_idx = min(len(out_frames_u8), int(seam_center_frame) + 2 * fps_i)
        sub_f = out_frames_u8[s_idx:e_idx]
        if len(sub_f) < 2:
            return
        h, w = int(sub_f.shape[1]) & ~1, int(sub_f.shape[2]) & ~1
        if h < 2 or w < 2:
            return
        if sub_f.shape[1] != h or sub_f.shape[2] != w:
            sub_f = sub_f[:, :h, :w, :]

        tmp_mp4 = f"{repaired_abs}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.mp4"
        try:
            with av.open(tmp_mp4, mode="w") as container:
                v_stream = container.add_stream("libx264", rate=fps_i)
                v_stream.width = w
                v_stream.height = h
                v_stream.pix_fmt = "yuv420p"
                v_stream.options = {"crf": "19", "preset": "fast"}
                a_stream = None
                if out_audio_np is not None and out_audio_np.shape[1] > 0:
                    n_ch = min(2, max(1, int(out_audio_np.shape[0])))
                    layout = "mono" if n_ch == 1 else "stereo"
                    a_stream = container.add_stream("aac", rate=int(sample_rate), layout=layout)
                for idx_f, fr in enumerate(sub_f):
                    vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")
                    vf.pts = idx_f
                    for pkt in v_stream.encode(vf):
                        container.mux(pkt)
                for pkt in v_stream.encode():
                    container.mux(pkt)
                if a_stream is not None and out_audio_np is not None:
                    s_samp = int(round((s_idx / float(fps_val)) * sample_rate))
                    e_samp = int(round((e_idx / float(fps_val)) * sample_rate))
                    sub_a = out_audio_np[:2, max(0, s_samp):min(out_audio_np.shape[1], e_samp)]
                    if sub_a.shape[1] > 0:
                        layout = "mono" if sub_a.shape[0] == 1 else "stereo"
                        chunk_sz = int(getattr(a_stream.codec_context, "frame_size", 0) or 1024)
                        pts_a = 0
                        for pos in range(0, sub_a.shape[1], chunk_sz):
                            seg = np.ascontiguousarray(sub_a[:, pos:pos + chunk_sz], dtype=np.float32)
                            if seg.shape[1] == 0:
                                continue
                            af = av.AudioFrame.from_ndarray(seg, format="fltp", layout=layout)
                            af.sample_rate = int(sample_rate)
                            af.pts = pts_a
                            pts_a += int(seg.shape[1])
                            for pkt in a_stream.encode(af):
                                container.mux(pkt)
                        for pkt in a_stream.encode():
                            container.mux(pkt)
            os.replace(tmp_mp4, repaired_abs)
        finally:
            if os.path.exists(tmp_mp4):
                try:
                    os.remove(tmp_mp4)
                except OSError:
                    pass

        with _SYNC_PROJ_LOCK:
            with open(pjson, "r", encoding="utf-8") as f:
                proj = json.load(f)
            rec = proj.setdefault("shots", {}).setdefault(str(int(shot_idx)), {"active_take_id": take_id, "locked_keep": False, "takes": {}})
            cur_sr = dict(rec.get("seam_repair") or {})
            cur_sr.update(repair_meta)
            cur_sr["take_id"] = take_id
            cur_sr["repaired_seam_rel"] = repaired_rel
            rec["seam_repair"] = cur_sr
            tmp = f"{pjson}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(proj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, pjson)
    except Exception:
        pass

try:
    from .fuzzpuppy_engine import format_bilingual_summary, merge_clips_fuzzpuppy
except ImportError:
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _fp_spec = _ilu.spec_from_file_location("acheng_fuzzpuppy_engine", _Path(__file__).resolve().parent / "fuzzpuppy_engine.py")
    _fp_mod = _ilu.module_from_spec(_fp_spec)
    _fp_spec.loader.exec_module(_fp_mod)
    format_bilingual_summary = _fp_mod.format_bilingual_summary
    merge_clips_fuzzpuppy = _fp_mod.merge_clips_fuzzpuppy


def _unpack_video_chunks(video) -> list:
    from comfy_api.latest import InputImpl

    if isinstance(video, InputImpl.VideoFromList):
        return list(video.videos)
    return [video]


def _resample_audio_np(audio_np: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio_np.shape[1] == 0:
        return audio_np.astype(np.float32, copy=False)
    dst_len = max(1, int(round(audio_np.shape[1] * float(dst_sr) / float(src_sr))))
    src_x = np.linspace(0.0, audio_np.shape[1] - 1, audio_np.shape[1], dtype=np.float64)
    dst_x = np.linspace(0.0, audio_np.shape[1] - 1, dst_len, dtype=np.float64)
    out = np.empty((audio_np.shape[0], dst_len), dtype=np.float32)
    for ch in range(audio_np.shape[0]):
        out[ch] = np.interp(dst_x, src_x, audio_np[ch]).astype(np.float32)
    return out


def _waveform_channel_count(audio_dict: Optional[dict]) -> int:
    if audio_dict is None or "waveform" not in audio_dict:
        return 0
    wf = audio_dict["waveform"]
    if wf.ndim >= 2:
        return int(wf.shape[-2])
    return 1


def _probe_chunk_audio_sig(chunk) -> tuple[Optional[int], Optional[int]]:
    """Inspect a VideoFromFile/VideoFromComponents audio stream without decoding video frames."""
    try:
        import av

        src = chunk.get_stream_source() if hasattr(chunk, "get_stream_source") else None
        if src is not None:
            with av.open(src) as container:
                for stream in container.streams.audio:
                    return int(stream.sample_rate), int(stream.channels)
            return None, None
    except Exception:
        pass
    try:
        comp = chunk.get_components()
        if comp.audio is not None and "waveform" in comp.audio:
            return int(comp.audio.get("sample_rate", 48000)), _waveform_channel_count(comp.audio)
    except Exception:
        pass
    return None, None


def _harmonize_chunk_audio(chunks: list, codec_enum) -> list:
    """Ensure all chunks share identical audio presence/sample_rate/channels so VideoFromList.save_to never crashes."""
    sigs = [_probe_chunk_audio_sig(c) for c in chunks]
    present = [(sr, ch) for sr, ch in sigs if sr is not None and ch is not None]
    if not present:
        return chunks
    target_sr = present[0][0]
    target_ch = max(ch for _, ch in present)
    if all(s == (target_sr, target_ch) for s in sigs):
        return chunks

    from comfy_api.latest import Types
    chunk_codec = Types.VideoCodec.H264 if codec_enum == Types.VideoCodec.AUTO else codec_enum
    out = []
    for chunk, sig in zip(chunks, sigs):
        if sig == (target_sr, target_ch):
            out.append(chunk)
            continue
        comp = chunk.get_components()
        frames_u8 = np.round(comp.images.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        if frames_u8.ndim == 4 and frames_u8.shape[-1] > 3:
            frames_u8 = frames_u8[..., :3]
        fps_frac = Fraction(comp.frame_rate) if comp.frame_rate else Fraction(24, 1)
        if comp.audio is not None:
            aud_np = _extract_audio_np(comp.audio, target_sr, target_ch)
        else:
            n_samps = max(1, int(round((len(frames_u8) / float(fps_frac)) * target_sr)))
            aud_np = np.zeros((target_ch, n_samps), dtype=np.float32)
        bit_depth = chunk.get_bit_depth() if hasattr(chunk, "get_bit_depth") else 8
        color_space = chunk.get_color_space() if hasattr(chunk, "get_color_space") else "sRGB"
        chroma_sub = chunk.get_chroma_subsampling() if hasattr(chunk, "get_chroma_subsampling") else None
        out.append(_make_video_chunk(frames_u8, fps_frac, aud_np, target_sr, bit_depth, color_space, chroma_sub, chunk_codec))
    return out


def _extract_audio_np(audio_dict: Optional[dict], target_sr: int, target_channels: int) -> Optional[np.ndarray]:
    if audio_dict is None or "waveform" not in audio_dict:
        return None
    wf = audio_dict["waveform"]
    if isinstance(wf, torch.Tensor):
        wf = wf.detach().cpu().float().numpy()
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim == 1:
        wf = wf[None, :]
    src_sr = int(audio_dict.get("sample_rate", target_sr))
    wf = _resample_audio_np(wf, src_sr, target_sr)
    if wf.shape[0] < target_channels:
        wf = np.repeat(wf[:1], target_channels, axis=0)
    elif wf.shape[0] > target_channels:
        wf = wf[:target_channels]
    return np.ascontiguousarray(wf, dtype=np.float32)


def _make_video_chunk(
    frames_u8: np.ndarray,
    fps_frac: Fraction,
    audio_np: Optional[np.ndarray],
    sample_rate: int,
    bit_depth: int,
    color_space: str,
    chroma_subsampling: Optional[str],
    codec_enum,
):
    from comfy_api.latest import InputImpl, Types

    img_t = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(dtype=torch.float32) / 255.0
    audio_dict = None
    if audio_np is not None and audio_np.shape[1] > 0:
        audio_dict = {
            "waveform": torch.from_numpy(np.ascontiguousarray(audio_np))[None, :, :],
            "sample_rate": int(sample_rate),
        }
    comp = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=img_t, audio=audio_dict, frame_rate=fps_frac),
        bit_depth=bit_depth,
        color_space=color_space,
        chroma_subsampling=chroma_subsampling,
    )
    # Encode into a self-contained stream chunk matching ComfyUI VideoFromList signature
    wrapped = InputImpl.VideoFromList([comp], codec=codec_enum)
    return wrapped.videos[0]


def _resolve_codec_enum(codec_str: str):
    from comfy_api.latest import Types

    try:
        return Types.VideoCodec(codec_str)
    except Exception:
        return Types.VideoCodec.AUTO


class AchengFuzzPuppyClipMerge:
    """Drop-in VIDEO+VIDEO seamless merger faithfully porting FuzzPuppy Clip Merge (`eI`+`eS`+`eT`+`V`+`eP`)."""

    CATEGORY = "Acheng/Video"
    FUNCTION = "merge"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "diagnostics")
    DESCRIPTION = (
        "FuzzPuppy Clip Merge 1:1 port for ComfyUI VIDEO streams: eI dual-gate hard-cut protection "
        "(structureError<0.38 & flowCoverage>0.08), eS 3-timestamp dense LK + 64-iter edge-aware diffusion "
        "+ 0.8s quintic release, rife422lite ONNX symmetric 3/5/7-frame bridge, and 30ms audio crossfade."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("VIDEO", {"tooltip": "First clip (or accumulated video ending with previous segment)."}),
                "video_b": ("VIDEO", {"tooltip": "Second clip to seamlessly merge after video_a."}),
                "mode": (["rife", "auto_align", "cut"], {"default": "rife", "tooltip": "rife: FuzzPuppy symmetric 3/5/7f rife422lite bridge; auto_align: FuzzPuppy eS dense LK + edge diffusion + 0.8s quintic release; cut: lossless hard cut."}),
                "rife_bridge_frames": (["3", "5", "7"], {"default": "5", "tooltip": "Symmetric bridge frame count across seam (FuzzPuppy default = 5)."}),
                "rife_backend": (["auto", "rife422lite_onnx", "rife426.pth", "rife49.pth", "rife47.pth"], {"default": "auto", "tooltip": "auto prefers local FuzzPuppy rife422lite_fp32.onnx on GPU, falling back to ComfyUI RIFE."}),
                "preserve_duration": ("BOOLEAN", {"default": False, "tooltip": "False = exact FuzzPuppy trim+4f overlap+30ms audio crossfade. True = zero frame_delta for strict external-audio MV sync."}),
                "max_search_sec": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Seam search window at A tail & B head in seconds (FuzzPuppy default = 0.5s)."}),
                "structure_threshold": ("FLOAT", {"default": 0.38, "min": 0.10, "max": 0.80, "step": 0.02, "tooltip": "FuzzPuppy eI max structureError gate (default 0.38). Higher = more permissive."}),
                "coverage_threshold": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01, "tooltip": "FuzzPuppy eI min bidirectional block-match coverage gate (default 0.08)."}),
                "release_seconds": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Release duration in seconds for auto_align quintic easing (FuzzPuppy default = 0.8s)."}),
            },
            "optional": {
                "is_continuation": ("BOOLEAN", {"default": True, "tooltip": "Wire ShotVault continuation flag here. False = instant zero-reencode lossless hard cut."}),
                "codec": (["auto", "h264", "av1", "prores"], {"default": "auto", "tooltip": "Codec for the merged seam chunk (auto matches ComfyUI CreateVideo H.264 signature)."}),
                "enable_repair": ("BOOLEAN", {"default": False, "tooltip": "Global repair switch. Default False = lossless original concat unless a shot is explicitly enabled in ShotVault Plan Director Console."}),
                "retry_policy": (["off", "once_ladder"], {"default": "off", "tooltip": "off = run selected mode once; once_ladder = if seam QC warns, test 1 bounded orthogonal fallback round and pick lowest defect."}),
                "ui_lang": (["zh", "en"], {"default": "zh", "tooltip": "Preferred language for summary field in diagnostics JSON (both summary_zh and summary_en are always included)."}),
                "shot_pipe": ("SHOT_PIPE", {"tooltip": "Optional ShotVault pipe. Syncs per-shot repair toggle/choice/params with ShotVault Plan Director Console so preview and merged video stay 100% consistent."}),
            },
        }

    def merge(
        self,
        video_a,
        video_b,
        mode: str = "rife",
        rife_bridge_frames: str = "5",
        rife_backend: str = "auto",
        preserve_duration: bool = False,
        max_search_sec: float = 0.5,
        structure_threshold: float = 0.38,
        coverage_threshold: float = 0.08,
        release_seconds: float = 0.8,
        is_continuation: bool = True,
        codec: str = "auto",
        enable_repair: bool = False,
        retry_policy: str = "off",
        ui_lang: str = "zh",
        shot_pipe: Any = None,
    ):
        from comfy_api.latest import InputImpl

        def _finalize_diag(rep: dict[str, Any]) -> str:
            if "summary_zh" not in rep or "summary_en" not in rep:
                sz, se = format_bilingual_summary(rep)
                rep["summary_zh"] = sz
                rep["summary_en"] = se
            rep["summary"] = rep["summary_en"] if ui_lang == "en" else rep["summary_zh"]
            return json.dumps(rep, ensure_ascii=False)

        codec_enum = _resolve_codec_enum(codec)
        chunks_a = _unpack_video_chunks(video_a)
        chunks_b = _unpack_video_chunks(video_b)

        # Fast-path: identical object (e.g. Shot 1 non-lazy ifElse branch)
        if video_a is video_b:
            report = {"mode": mode, "enable_repair": bool(enable_repair), "applied": "passthrough_same_video", "safe": False, "reason": "same_video_object_passthrough"}
            return (video_b, _finalize_diag(report))

        # Resolve per-shot Director Console override when shot_pipe is wired (Single Source of Truth!)
        proj_root, shot_idx, take_id, sr_override = _read_shot_repair_override(shot_pipe)
        eff_enable = bool(enable_repair)
        eff_mode = mode
        eff_bridge = int(rife_bridge_frames)
        eff_preserve = bool(preserve_duration)
        eff_struct = float(structure_threshold)
        eff_cov = float(coverage_threshold)
        eff_retry = retry_policy
        control_source = "node_global"

        if isinstance(sr_override, dict) and "enabled" in sr_override:
            control_source = "director_console_per_shot"
            want_repaired = bool(sr_override.get("enabled", False)) and str(sr_override.get("choice") or "repaired") == "repaired"
            eff_enable = want_repaired
            if want_repaired:
                eff_mode = str(sr_override.get("mode") or eff_mode)
                eff_bridge = int(sr_override.get("rife_bridge_frames") or eff_bridge)
                eff_preserve = bool(sr_override.get("preserve_duration", eff_preserve))
                eff_struct = float(sr_override.get("structure_threshold", eff_struct))
                eff_cov = float(sr_override.get("coverage_threshold", eff_cov))
                eff_retry = str(sr_override.get("retry_policy") or eff_retry)

        if not eff_enable:
            safe_chunks = _harmonize_chunk_audio(chunks_a + chunks_b, codec_enum)
            merged = InputImpl.VideoFromList(safe_chunks, codec=codec_enum)
            report = {
                "mode": eff_mode,
                "enable_repair": False,
                "control_source": control_source,
                "shot_index": shot_idx,
                "applied": "manual_disabled_original",
                "safe": False,
                "reason": "director_console_original_selected" if control_source == "director_console_per_shot" else "repair_disabled_by_user",
                "chunks": len(merged.videos),
                "reencoded_chunks": 0,
            }
            return (merged, _finalize_diag(report))

        if not is_continuation or eff_mode == "cut":
            safe_chunks = _harmonize_chunk_audio(chunks_a + chunks_b, codec_enum)
            merged = InputImpl.VideoFromList(safe_chunks, codec=codec_enum)
            report = {
                "mode": eff_mode,
                "enable_repair": True,
                "control_source": control_source,
                "shot_index": shot_idx,
                "applied": "lossless_hard_cut",
                "safe": False,
                "reason": "not_continuation_planned_cut" if not is_continuation else "explicit_cut_mode",
                "chunks": len(merged.videos),
                "reencoded_chunks": 0,
            }
            return (merged, _finalize_diag(report))

        prefix_chunks = chunks_a[:-1]
        tail_a = chunks_a[-1]
        head_b = chunks_b[0]
        suffix_chunks = chunks_b[1:]

        # Decode ONLY boundary pair (tail_a, head_b); prefix_chunks & suffix_chunks are NEVER decoded or re-encoded
        comp_a = tail_a.get_components()
        comp_b = head_b.get_components()

        frames_a_u8 = np.round(comp_a.images.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        frames_b_u8 = np.round(comp_b.images.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        if frames_a_u8.ndim == 4 and frames_a_u8.shape[-1] > 3:
            frames_a_u8 = frames_a_u8[..., :3]
        if frames_b_u8.ndim == 4 and frames_b_u8.shape[-1] > 3:
            frames_b_u8 = frames_b_u8[..., :3]
        fps_frac = Fraction(comp_a.frame_rate) if comp_a.frame_rate else Fraction(24, 1)
        fps_val = float(fps_frac)

        # Unify audio parameters across prefix, tail_a, head_b, and suffix so VideoFromList.save_to always stays compatible
        outer_sigs = [_probe_chunk_audio_sig(c) for c in (prefix_chunks + suffix_chunks)]
        outer_present = [(sr, ch) for sr, ch in outer_sigs if sr is not None and ch is not None]
        has_any_audio = (comp_a.audio is not None) or (comp_b.audio is not None) or bool(outer_present)
        sample_rate = 48000
        n_channels = 2
        if has_any_audio:
            sr_a = int(comp_a.audio["sample_rate"]) if comp_a.audio is not None else None
            sr_b = int(comp_b.audio["sample_rate"]) if comp_b.audio is not None else None
            sample_rate = outer_present[0][0] if outer_present else (sr_a if sr_a is not None else (sr_b if sr_b is not None else 48000))
            ch_a = _waveform_channel_count(comp_a.audio)
            ch_b = _waveform_channel_count(comp_b.audio)
            ch_outer = max((ch for _, ch in outer_present), default=0)
            n_channels = max(1, ch_a, ch_b, ch_outer)
        audio_a_np = _extract_audio_np(comp_a.audio, sample_rate, n_channels) if has_any_audio else None
        audio_b_np = _extract_audio_np(comp_b.audio, sample_rate, n_channels) if has_any_audio else None

        out_frames_u8, out_audio_np, report = merge_clips_fuzzpuppy(
            frames_a_u8=frames_a_u8,
            frames_b_u8=frames_b_u8,
            fps=fps_val,
            mode=eff_mode,
            rife_bridge_frames=eff_bridge,
            rife_backend=rife_backend,
            preserve_duration=eff_preserve,
            max_search_sec=float(max_search_sec),
            overlap_frames=4,
            structure_threshold=eff_struct,
            coverage_threshold=eff_cov,
            release_seconds=float(release_seconds),
            audio_a=audio_a_np,
            audio_b=audio_b_np,
            sample_rate=sample_rate,
            retry_policy=eff_retry,
        )
        report["enable_repair"] = True
        report["control_source"] = control_source
        report["shot_index"] = shot_idx

        # If eI dual-gate determined this is a hard cut / unreliable correspondence -> ZERO-REENCODE passthrough!
        if not report.get("safe", False) or report.get("applied") == "hard_cut_fallback":
            safe_chunks = _harmonize_chunk_audio(prefix_chunks + [tail_a, head_b] + suffix_chunks, codec_enum)
            merged = InputImpl.VideoFromList(safe_chunks, codec=codec_enum)
            report["chunks"] = len(merged.videos)
            report["reencoded_chunks"] = 0
            return (merged, _finalize_diag(report))

        # Sync repaired 3s preview & exact report back to ShotVault project.json so Director Console always matches merged output!
        if proj_root and shot_idx is not None and take_id:
            if "rife_left_a_idx" in report and report["rife_left_a_idx"] is not None:
                seam_center = int(report["rife_left_a_idx"])
            else:
                trim_a = float(report.get("trimA_frames") or 0)
                ov = float(report.get("overlap_frames") or 0)
                seam_center = max(0, int(round(len(frames_a_u8) - trim_a - ov * 0.5)))
            _sync_repair_preview_to_vault(
                proj_root=proj_root,
                shot_idx=shot_idx,
                take_id=take_id,
                out_frames_u8=out_frames_u8,
                out_audio_np=out_audio_np,
                fps_val=fps_val,
                sample_rate=sample_rate,
                seam_center_frame=seam_center,
                repair_meta={
                    "enabled": True,
                    "choice": "repaired",
                    "mode": eff_mode,
                    "rife_bridge_frames": eff_bridge,
                    "preserve_duration": eff_preserve,
                    "structure_threshold": eff_struct,
                    "coverage_threshold": eff_cov,
                    "retry_policy": eff_retry,
                    "report": report,
                },
            )

        bit_depth = tail_a.get_bit_depth() if hasattr(tail_a, "get_bit_depth") else 8
        color_space = tail_a.get_color_space() if hasattr(tail_a, "get_color_space") else "sRGB"
        chroma_sub = tail_a.get_chroma_subsampling() if hasattr(tail_a, "get_chroma_subsampling") else None
        # Ensure encoded chunks use H.264 when codec is auto so signatures stay consistent with CreateVideo(h264)
        from comfy_api.latest import Types
        chunk_codec = Types.VideoCodec.H264 if codec_enum == Types.VideoCodec.AUTO else codec_enum

        # Split strictly BEFORE the seam zone so earlier frames of tail_a are finalized once and NEVER re-encoded on future shots
        n_a = len(frames_a_u8)
        safe_margin = max(24, int(math.ceil(max_search_sec * fps_val)) + 12)
        split_idx = n_a - safe_margin
        new_chunks = []
        if split_idx >= 16 and (len(out_frames_u8) - split_idx) >= 16:
            split_samp = int(round((split_idx / float(fps_frac)) * sample_rate))
            aud_part1 = out_audio_np[:, :split_samp] if out_audio_np is not None else None
            aud_part2 = out_audio_np[:, split_samp:] if out_audio_np is not None else None
            chunk_finalized_a = _make_video_chunk(
                out_frames_u8[:split_idx], fps_frac, aud_part1, sample_rate, bit_depth, color_space, chroma_sub, chunk_codec
            )
            chunk_seam_and_b = _make_video_chunk(
                out_frames_u8[split_idx:], fps_frac, aud_part2, sample_rate, bit_depth, color_space, chroma_sub, chunk_codec
            )
            new_chunks = [chunk_finalized_a, chunk_seam_and_b]
        else:
            chunk_all = _make_video_chunk(
                out_frames_u8, fps_frac, out_audio_np, sample_rate, bit_depth, color_space, chroma_sub, chunk_codec
            )
            new_chunks = [chunk_all]

        final_chunks = _harmonize_chunk_audio(prefix_chunks + new_chunks + suffix_chunks, chunk_codec)
        merged = InputImpl.VideoFromList(final_chunks, codec=chunk_codec)
        report["chunks"] = len(merged.videos)
        report["untouched_prefix_chunks"] = len(prefix_chunks)
        report["split_idx"] = int(split_idx) if len(new_chunks) == 2 else 0
        return (merged, _finalize_diag(report))
