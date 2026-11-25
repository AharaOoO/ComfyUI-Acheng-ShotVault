"""Single-pass f32le PCM master assembler + lossless video concat + DaVinci Resolve FCPXML v1.9 exporter."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..compat import ensure_inside_root, resample_audio, resolve_ffmpeg_exe
from .grid_math import DELIVERY_AUDIO_SR, FPS_DEFAULT
from .lineage_dag import LineageDAG
from .vault_store import VaultStore


def _decode_mp4_audio_48k(mp4_path: str) -> Optional[torch.Tensor]:
    import av  # type: ignore

    chunks: list[torch.Tensor] = []
    src_sr = DELIVERY_AUDIO_SR
    with av.open(mp4_path) as container:
        a_stream = next((s for s in container.streams if s.type == "audio"), None)
        if a_stream is None:
            return None
        src_sr = int(a_stream.sample_rate or DELIVERY_AUDIO_SR)
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            nb_ch = int(getattr(frame.layout, "nb_channels", arr.shape[0] if arr.ndim > 1 else 1) or 1)
            if not frame.format.is_planar and nb_ch > 1:
                arr = arr.reshape(-1, nb_ch).T
            elif arr.ndim == 1:
                arr = arr[None, :]
            t = torch.from_numpy(np.ascontiguousarray(arr)).float()
            if t.abs().max() > 1.5:
                t = t / 32768.0
            if t.shape[0] == 1:
                t = t.repeat(2, 1)
            elif t.shape[0] > 2:
                t = t[:2]
            chunks.append(t)
    if not chunks:
        return None
    return resample_audio(torch.cat(chunks, dim=1), src_sr, DELIVERY_AUDIO_SR)


def _count_video_frames(mp4_path: str) -> int:
    import av  # type: ignore

    count = 0
    with av.open(mp4_path) as container:
        v_stream = next((s for s in container.streams if s.type == "video"), None)
        if v_stream is None:
            return 0
        if v_stream.frames and int(v_stream.frames) > 0:
            return int(v_stream.frames)
        for _ in container.decode(video=0):
            count += 1
    return count


def _apply_boundary_declick(segments: list[torch.Tensor], crossfade_ms: float) -> torch.Tensor:
    """Concatenate 48kHz `[2, S_i]` segments preserving exact sample sum with anti-click ramps."""
    if not segments:
        return torch.zeros((2, DELIVERY_AUDIO_SR), dtype=torch.float32)
    if len(segments) == 1 or float(crossfade_ms) <= 0.0:
        return torch.cat(segments, dim=1)

    fade_half = max(0, round(float(crossfade_ms) * 0.5 / 1000.0 * DELIVERY_AUDIO_SR))
    processed = [s.clone() for s in segments]
    for i in range(len(processed) - 1):
        a = processed[i]
        b = processed[i + 1]
        n_f = min(fade_half, a.shape[1] // 4, b.shape[1] // 4)
        if n_f > 1:
            t = torch.linspace(0.0, 1.0, steps=n_f, dtype=torch.float32).unsqueeze(0)
            a[:, -n_f:] *= (1.0 - t)
            b[:, :n_f] *= t
    return torch.cat(processed, dim=1)


def _export_fcpxml(
    clips: list[dict[str, Any]],
    project_name: str,
    fps: int,
    width: int,
    height: int,
    out_fcpxml: str,
) -> str:
    """Generate clean DaVinci Resolve FCPXML v1.9 with exact integer-frame rational timing."""
    total_frames = sum(int(c["frames"]) for c in clips)
    scale = int(fps) * 100

    def fd(frames: int) -> str:
        return f"{int(frames) * 100}/{scale}s"

    root = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        name=f"FFVideoFormat{height}p{fps}",
        frameDuration=f"100/{scale}s",
        width=str(width),
        height=str(height),
    )
    for idx, clip in enumerate(clips, start=2):
        rid = f"r{idx}"
        clip["asset_ref"] = rid
        uri = "file:///" + os.path.abspath(clip["path"]).replace("\\", "/")
        asset = ET.SubElement(
            resources,
            "asset",
            id=rid,
            name=os.path.basename(clip["path"]),
            src=uri,
            start="0s",
            duration=fd(clip["frames"]),
            hasVideo="1",
            hasAudio="1",
            format="r1",
        )
        ET.SubElement(asset, "media-rep", kind="original-media", src=uri)

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=project_name)
    project = ET.SubElement(event, "project", name=f"{project_name}_Master")
    seq = ET.SubElement(project, "sequence", format="r1", duration=fd(total_frames), tcStart="0s", tcFormat="NDF")
    spine = ET.SubElement(seq, "spine")
    cursor = 0
    for clip in clips:
        ET.SubElement(
            spine,
            "asset-clip",
            name=clip["label"],
            ref=clip["asset_ref"],
            offset=fd(cursor),
            start="0s",
            duration=fd(clip["frames"]),
            tcFormat="NDF",
        )
        cursor += int(clip["frames"])

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_fcpxml, encoding="utf-8", xml_declaration=True)
    return out_fcpxml


def assemble_master(
    store: VaultStore,
    shots_plan: list[dict[str, Any]],
    fps: int = FPS_DEFAULT,
    stale_policy: str = "defer_until_complete",
    audio_source_policy: str = "auto_prefer_reference",
    master_audio_override: Optional[dict] = None,
    audio_crossfade_ms: float = 15.0,
    post_mux_offset_ms: float = 0.0,
    prepend_seed_video: bool = True,
    export_fcpxml: bool = True,
    master_filename: str = "",
    has_pending_queue: bool = False,
) -> tuple[str, str, str]:
    """Concatenate video streams losslessly and mux 48kHz in-memory PCM in a single FFmpeg pass."""
    project = store.load_project()
    dag = LineageDAG.evaluate(shots_plan, project)
    blocking = [d for d in dag if d["status"] in ("UNRENDERED", "STALE")]
    unrendered = [d["shot_index"] for d in dag if d["status"] == "UNRENDERED"]
    stale = [d["shot_index"] for d in dag if d["status"] == "STALE"]
    warnings = [d for d in dag if d["status"] in ("LOCKED_KEEP", "STALE", "UNRENDERED") or d["prompt_edited"]]

    if stale_policy == "defer_until_complete" and (has_pending_queue or unrendered or stale):
        deferred = {
            "status": "deferred",
            "has_pending_queue": bool(has_pending_queue),
            "unrendered_shots": unrendered,
            "stale_shots": stale,
            "message": "自动等待剩余或标红镜头跑完后再单遍封装总成片。",
            "message_en": "Deferred master assembly until remaining/STALE shots finish.",
        }
        return "", "", json.dumps(deferred, ensure_ascii=False, indent=2)

    if blocking and stale_policy == "block_and_report":
        summary = ", ".join(f"Shot {d['shot_index']} ({d['status']})" for d in blocking)
        raise ValueError(f"ShotVaultAssemble blocked by unrendered/STALE shots: {summary}")

    clips: list[dict[str, Any]] = []
    seed_shot = project.get("seed_shot") or {}
    if prepend_seed_video and seed_shot.get("take_id"):
        seed_path = ensure_inside_root(store.project_root, str(seed_shot.get("video_rel_path") or "shots/shot_0000_seed.mp4"))
        if os.path.isfile(seed_path):
            clips.append({
                "label": "shot_0000_seed",
                "take_id": str(seed_shot["take_id"]),
                "path": seed_path,
                "frames": int(seed_shot.get("frames") or _count_video_frames(seed_path) or 1),
                "width": 1344,
                "height": 768,
            })

    shots_state = project.get("shots") or {}
    for entry in dag:
        tid = entry.get("active_take_id") or ""
        if not tid:
            continue
        rec = (shots_state.get(str(entry["shot_index"])) or {}).get("takes", {}).get(tid) or {}
        rel = str(rec.get("video_rel_path") or f"shots/{tid}.mp4")
        clip_path = ensure_inside_root(store.project_root, rel)
        if not os.path.isfile(clip_path):
            if stale_policy == "block_and_report":
                raise FileNotFoundError(f"ShotVaultAssemble: missing clip video {clip_path!r}")
            continue
        clips.append({
            "label": entry["shot_id"],
            "shot_index": int(entry["shot_index"]),
            "take_id": tid,
            "path": clip_path,
            "frames": int(rec.get("keep_frames") or _count_video_frames(clip_path) or 1),
            "width": int(rec.get("width") or 1344),
            "height": int(rec.get("height") or 768),
            "seam_repair": dict((shots_state.get(str(entry["shot_index"])) or {}).get("seam_repair") or {}),
        })

    if not clips:
        raise ValueError("ShotVaultAssemble: no rendered takes found to assemble.")

    proj_name = str(project.get("project_name") or os.path.basename(store.project_root))
    base_name = master_filename.strip() or f"{proj_name}_master"
    master_mp4 = os.path.join(store.master_dir, f"{base_name}.mp4")
    fcpxml_path = os.path.join(store.master_dir, f"{base_name}.fcpxml")
    ffmpeg_exe = resolve_ffmpeg_exe()

    applied_seam_repairs: list[dict[str, Any]] = []
    # Match source codec (prores_ks vs h264) so FFmpeg concat -c:v copy never hits codec/pix_fmt mismatch
    is_prores = any(str(c["path"]).lower().endswith(".mov") for c in clips)
    merge_codec = "prores_ks" if is_prores else "h264"
    merge_ext = "mov" if is_prores else "mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Apply user-enabled per-shot FuzzPuppy seam repairs ONLY to explicitly approved boundaries
        active_clips: list[dict[str, Any]] = []
        for idx, clip in enumerate(clips):
            sr = clip.get("seam_repair") or {}
            want_repair = (
                idx > 0
                and bool(sr.get("enabled", False))
                and str(sr.get("choice") or "repaired") == "repaired"
                and str(sr.get("mode") or "rife") != "cut"
            )
            if want_repair and active_clips:
                from .seam_repair_bridge import apply_repaired_seam_to_clip_pair

                prev_clip = active_clips[-1]
                merged_path = os.path.join(tmpdir, f"seam_merged_{idx:04d}.{merge_ext}")
                new_frames, rep = apply_repaired_seam_to_clip_pair(
                    prev_clip["path"], clip["path"], sr, merged_path, fps=int(fps), codec=merge_codec
                )
                active_clips[-1] = {
                    **prev_clip,
                    "label": f"{prev_clip['label']}+{clip['label']}",
                    "path": merged_path,
                    "frames": int(new_frames),
                    "take_id": "",
                }
                applied_seam_repairs.append({"shot_index": clip.get("shot_index"), "applied": rep.get("applied"), "backend": rep.get("backend"), "frame_delta": rep.get("frame_delta", 0)})
            else:
                active_clips.append(clip)

        total_frames = sum(int(c["frames"]) for c in active_clips)
        total_samples = max(1, round(total_frames / float(fps) * DELIVERY_AUDIO_SR))

        has_master_override = (
            audio_source_policy == "master_override"
            and isinstance(master_audio_override, dict)
            and isinstance(master_audio_override.get("waveform"), torch.Tensor)
        )
        can_single_clip_fast_copy = (
            len(active_clips) == 1
            and not applied_seam_repairs
            and abs(float(post_mux_offset_ms)) < 1e-4
            and not has_master_override
            and str(active_clips[0]["path"]).lower().endswith(".mp4")
            and os.path.isfile(active_clips[0]["path"])
        )

        if can_single_clip_fast_copy:
            import shutil

            tmp_master_mp4 = os.path.join(store.master_dir, f"{base_name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}.mp4")
            try:
                shutil.copy2(active_clips[0]["path"], tmp_master_mp4)
                os.replace(tmp_master_mp4, master_mp4)
            finally:
                if os.path.exists(tmp_master_mp4):
                    try:
                        os.remove(tmp_master_mp4)
                    except OSError:
                        pass
            audio_samples_48k = int(total_samples)
        else:
            if has_master_override:
                mw = master_audio_override["waveform"][0].detach().cpu().float()
                if mw.shape[0] == 1:
                    mw = mw.repeat(2, 1)
                elif mw.shape[0] > 2:
                    mw = mw[:2]
                m48 = resample_audio(mw, int(master_audio_override.get("sample_rate", 48000)), DELIVERY_AUDIO_SR)
                if m48.shape[1] < total_samples:
                    m48 = F.pad(m48, (0, total_samples - m48.shape[1]))
                full_pcm = m48[:, :total_samples]
            else:
                audio_segs: list[torch.Tensor] = []
                for clip in active_clips:
                    need_s = max(1, round(clip["frames"] / float(fps) * DELIVERY_AUDIO_SR))
                    wav = None
                    if clip.get("take_id"):
                        ctx = store.load_take_context(clip["take_id"])
                        if ctx is not None and "delivery_audio_48k" in ctx["tensors"]:
                            wav = ctx["tensors"]["delivery_audio_48k"].float()
                    if wav is None:
                        wav = _decode_mp4_audio_48k(clip["path"])
                    if wav is None:
                        wav = torch.zeros((2, need_s), dtype=torch.float32)
                    if wav.shape[0] == 1:
                        wav = wav.repeat(2, 1)
                    elif wav.shape[0] > 2:
                        wav = wav[:2]
                    if wav.shape[1] < need_s:
                        wav = F.pad(wav, (0, need_s - wav.shape[1]))
                    else:
                        wav = wav[:, :need_s]
                    audio_segs.append(wav)
                full_pcm = _apply_boundary_declick(audio_segs, float(audio_crossfade_ms))
                if full_pcm.shape[1] < total_samples:
                    full_pcm = F.pad(full_pcm, (0, total_samples - full_pcm.shape[1]))
                else:
                    full_pcm = full_pcm[:, :total_samples]

            offset_samples = round(float(post_mux_offset_ms) / 1000.0 * DELIVERY_AUDIO_SR)
            if offset_samples > 0:
                full_pcm = torch.cat([torch.zeros((2, offset_samples), dtype=torch.float32), full_pcm], dim=1)[:, :total_samples]
            elif offset_samples < 0:
                shift = min(-offset_samples, full_pcm.shape[1])
                full_pcm = torch.cat([full_pcm[:, shift:], torch.zeros((2, shift), dtype=torch.float32)], dim=1)

            list_txt = os.path.join(tmpdir, "concat_list.txt")
            video_only = os.path.join(tmpdir, "video_only.mp4")
            with open(list_txt, "w", encoding="utf-8") as f:
                for clip in active_clips:
                    escaped = os.path.abspath(clip["path"]).replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            c_proc = subprocess.run(
                [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_txt, "-an", "-c:v", "copy", video_only],
                capture_output=True,
            )
            if c_proc.returncode != 0:
                raise RuntimeError(f"ShotVault FFmpeg concat failed: {c_proc.stderr.decode('utf-8', 'replace')}")

            pcm_bytes = np.ascontiguousarray(full_pcm.transpose(0, 1).numpy(), dtype="<f4").tobytes()
            tmp_master_mp4 = os.path.join(store.master_dir, f"{base_name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}.mp4")
            mux_cmd = [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_only,
                "-f", "f32le", "-ar", str(DELIVERY_AUDIO_SR), "-ac", "2", "-i", "pipe:0",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart",
                tmp_master_mp4,
            ]
            try:
                m_proc = subprocess.run(mux_cmd, input=pcm_bytes, capture_output=True)
                if m_proc.returncode != 0:
                    raise RuntimeError(f"ShotVault single-pass PCM mux failed: {m_proc.stderr.decode('utf-8', 'replace')}")
                os.replace(tmp_master_mp4, master_mp4)
            finally:
                if os.path.exists(tmp_master_mp4):
                    try:
                        os.remove(tmp_master_mp4)
                    except OSError:
                        pass
            audio_samples_48k = int(full_pcm.shape[1])

    written_fcpxml = ""
    if export_fcpxml:
        fcpxml_clips = (
            [{"label": f"{proj_name}_repaired_master", "path": master_mp4, "frames": total_frames, "width": clips[0]["width"], "height": clips[0]["height"]}]
            if applied_seam_repairs
            else clips
        )
        written_fcpxml = _export_fcpxml(
            fcpxml_clips, proj_name, int(fps), clips[0]["width"], clips[0]["height"], fcpxml_path
        )

    # Atomically persist lightweight sidecar manifest so UI knows exact takes & seam repairs baked into this master
    manifest_shots: dict[str, Any] = {}
    for c in clips:
        s_idx = c.get("shot_index")
        if s_idx is None:
            continue
        sr = c.get("seam_repair") or {}
        sr_on = bool(sr.get("enabled", False)) and str(sr.get("choice") or "repaired") == "repaired" and str(sr.get("mode") or "rife") != "cut"
        manifest_shots[str(int(s_idx))] = {
            "take_id": str(c.get("take_id") or ""),
            "seam_repair_sig": (
                f"{sr.get('mode','rife')}:{sr.get('rife_bridge_frames',5)}:{bool(sr.get('preserve_duration',False))}:{sr.get('take_id','')}"
                if sr_on
                else "original"
            ),
        }
    manifest_path = os.path.join(store.master_dir, f"{base_name}.manifest.json")
    tmp_manifest = f"{manifest_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}"
    try:
        with open(tmp_manifest, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "master_file": os.path.basename(master_mp4),
                    "fcpxml_file": os.path.basename(written_fcpxml) if written_fcpxml else "",
                    "exported_at": time.time(),
                    "total_clips": len(clips),
                    "total_frames": total_frames,
                    "shots": manifest_shots,
                },
                mf,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp_manifest, manifest_path)
    except Exception:
        if os.path.exists(tmp_manifest):
            try:
                os.remove(tmp_manifest)
            except OSError:
                pass

    report = {
        "master_mp4": master_mp4,
        "fcpxml_path": written_fcpxml,
        "total_clips": len(clips),
        "total_frames": total_frames,
        "expected_duration_sec": round(total_frames / float(fps), 4),
        "audio_samples_48k": int(audio_samples_48k),
        "single_clip_fast_copy": bool(can_single_clip_fast_copy),
        "post_mux_offset_ms": float(post_mux_offset_ms),
        "applied_seam_repairs": applied_seam_repairs,
        "warnings": warnings,
    }
    return master_mp4, written_fcpxml, json.dumps(report, ensure_ascii=False, indent=2)
