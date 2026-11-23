"""External Shot 0 video ingestion, arbitrary tail trim, dual-rate audio standardization, and archival."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..compat import resample_audio
from .grid_math import AUDIO_VAE_SR, DELIVERY_AUDIO_SR, FPS_DEFAULT, pad_audio_800
from .vault_store import VaultStore


def _load_av_media(video_path: str, max_trim_end_sec: float = 0.0, target_fps: int = FPS_DEFAULT) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Decode RGB frames and stereo audio from MP4/MOV via PyAV, trimming past `max_trim_end_sec` if > 0."""
    import av  # type: ignore

    frames_rgb: list[np.ndarray] = []
    audio_chunks: list[torch.Tensor] = []
    src_sr = DELIVERY_AUDIO_SR
    src_fps = float(target_fps)

    with av.open(video_path) as container:
        v_stream = next((s for s in container.streams if s.type == "video"), None)
        if v_stream is None:
            raise ValueError(f"ShotVault: no video stream in seed video {video_path!r}")
        if v_stream.average_rate:
            src_fps = float(v_stream.average_rate)
        max_frames = int(round(max_trim_end_sec * src_fps)) if max_trim_end_sec > 0 else None

        for frame in container.decode(video=0):
            if max_frames is not None and len(frames_rgb) >= max_frames:
                break
            frames_rgb.append(frame.to_ndarray(format="rgb24"))

    # Audio stream MUST open its own container handle because decoding video=0
    # to EOF advances the shared demuxer past interleaved audio packets.
    with av.open(video_path) as a_container:
        a_stream = next((s for s in a_container.streams if s.type == "audio"), None)
        if a_stream is not None:
            src_sr = int(a_stream.sample_rate or DELIVERY_AUDIO_SR)
            max_samples = int(round(max_trim_end_sec * src_sr)) if max_trim_end_sec > 0 else None
            total_samples = 0
            for frame in a_container.decode(audio=0):
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
                if max_samples is not None and total_samples + t.shape[1] > max_samples:
                    t = t[:, : max(0, max_samples - total_samples)]
                if t.shape[1] > 0:
                    audio_chunks.append(t)
                    total_samples += int(t.shape[1])
                if max_samples is not None and total_samples >= max_samples:
                    break

    if not frames_rgb:
        raise ValueError(f"ShotVault: decoded 0 video frames from {video_path!r}")
    vid_t = torch.from_numpy(np.stack(frames_rgb, axis=0)).float().div_(255.0)
    if abs(src_fps - float(target_fps)) > 0.25 and vid_t.shape[0] > 1:
        new_n = max(1, round(vid_t.shape[0] * float(target_fps) / src_fps))
        idx = torch.linspace(0, vid_t.shape[0] - 1, steps=new_n).round().long()
        vid_t = vid_t[idx]

    target_samples_48k = max(1, round(vid_t.shape[0] / float(target_fps) * DELIVERY_AUDIO_SR))
    if audio_chunks:
        wav = torch.cat(audio_chunks, dim=1)
        wav_48k = resample_audio(wav, src_sr, DELIVERY_AUDIO_SR)
        if wav_48k.shape[1] < target_samples_48k:
            wav_48k = F.pad(wav_48k, (0, target_samples_48k - wav_48k.shape[1]))
        else:
            wav_48k = wav_48k[:, :target_samples_48k]
    else:
        wav_48k = torch.zeros((2, target_samples_48k), dtype=torch.float32)

    return vid_t.contiguous(), wav_48k.contiguous(), float(target_fps)


def _write_seed_mp4_and_thumb(
    store: VaultStore,
    frames_hwc: torch.Tensor,
    wav_48k: torch.Tensor,
    fps: int,
) -> tuple[str, str]:
    from .seam_qc import write_jpg_thumb, write_mp4_with_pcm

    mp4_rel = "shots/shot_0000_seed.mp4"
    thumb_rel = "_vault/thumb_shot_0000_seed.jpg"
    write_mp4_with_pcm(
        frames_hwc,
        wav_48k,
        DELIVERY_AUDIO_SR,
        fps,
        os.path.join(store.project_root, mp4_rel),
    )
    write_jpg_thumb(frames_hwc[-1], os.path.join(store.project_root, thumb_rel))
    return mp4_rel, thumb_rel


def _encode_seed_tail_latent(
    frames: torch.Tensor,
    wav_48k: torch.Tensor,
    fps: int,
    vae=None,
    audio_vae=None,
) -> tuple[Optional[dict], Optional[int]]:
    """Optionally pre-encode last valid 39f/22f tail of Shot 0 into AV NestedTensor latent."""
    if vae is None or frames.shape[0] < 22:
        return None, None
    tail_n = 39 if int(frames.shape[0]) >= 39 else 22
    try:
        import comfy.nested_tensor  # type: ignore

        v_tail = vae.encode(frames[-tail_n:].float())
        if audio_vae is not None:
            w_32k = resample_audio(wav_48k.detach().cpu().float(), DELIVERY_AUDIO_SR, AUDIO_VAE_SR)
            need_s = max(800, round(tail_n / float(fps) * AUDIO_VAE_SR))
            if w_32k.shape[1] < need_s:
                w_32k = F.pad(w_32k, (need_s - int(w_32k.shape[1]), 0))
            else:
                w_32k = w_32k[:, -need_s:]
            a_inp = pad_audio_800(w_32k.unsqueeze(0)).movedim(1, -1)
            a_tail = audio_vae.encode(a_inp)
        else:
            a_t = max(1, round(tail_n * (40.0 / float(fps))))
            a_tail = torch.zeros((1, 32, 2, a_t), dtype=torch.float32)
        return {"samples": comfy.nested_tensor.NestedTensor((v_tail, a_tail))}, tail_n
    except Exception:
        return None, None


def ingest_seed_video(
    store: VaultStore,
    seed_video_path: str = "",
    seed_trim_end_sec: float = 0.0,
    width: int = 1344,
    height: int = 768,
    fps: int = FPS_DEFAULT,
    reuse_seed_images: Optional[torch.Tensor] = None,
    reuse_seed_audio: Optional[dict] = None,
    vae=None,
    audio_vae=None,
) -> Optional[dict[str, Any]]:
    """Ingest external seed video (Shot 0) or wired image/audio batch, trim tail, normalize, and store `.h3ctx`."""
    if reuse_seed_images is not None and isinstance(reuse_seed_images, torch.Tensor) and reuse_seed_images.shape[0] > 0:
        frames = reuse_seed_images[..., :3].detach().cpu().float().clamp(0.0, 1.0)
        if float(seed_trim_end_sec) > 0:
            max_f = max(1, round(float(seed_trim_end_sec) * fps))
            frames = frames[:max_f]
        target_len_48k = max(1, round(frames.shape[0] / float(fps) * DELIVERY_AUDIO_SR))
        if isinstance(reuse_seed_audio, dict) and isinstance(reuse_seed_audio.get("waveform"), torch.Tensor):
            w = reuse_seed_audio["waveform"][0].detach().cpu().float()
            if w.shape[0] == 1:
                w = w.repeat(2, 1)
            elif w.shape[0] > 2:
                w = w[:2]
            w48 = resample_audio(w, int(reuse_seed_audio.get("sample_rate", 48000)), DELIVERY_AUDIO_SR)
            if w48.shape[1] < target_len_48k:
                w48 = F.pad(w48, (0, target_len_48k - w48.shape[1]))
            wav_48k = w48[:, :target_len_48k].contiguous()
        else:
            wav_48k = torch.zeros((2, target_len_48k), dtype=torch.float32)
    elif seed_video_path and os.path.isfile(seed_video_path):
        frames, wav_48k, _ = _load_av_media(seed_video_path, float(seed_trim_end_sec), int(fps))
    else:
        return None

    if int(frames.shape[1]) != int(height) or int(frames.shape[2]) != int(width):
        chw = frames.movedim(-1, 1)
        chw = F.interpolate(chw, size=(int(height), int(width)), mode="bilinear", align_corners=False)
        frames = chw.movedim(1, -1).contiguous()

    mp4_rel, thumb_rel = _write_seed_mp4_and_thumb(store, frames, wav_48k, int(fps))
    sampled_lat, lat_tail_f = _encode_seed_tail_latent(frames, wav_48k, int(fps), vae=vae, audio_vae=audio_vae)
    meta = store.save_take_context(
        take_id="shot_0000_seed",
        shot_index=0,
        clean_images=frames,
        audio_waveform_48k=wav_48k,
        audio_sr=DELIVERY_AUDIO_SR,
        fps=int(fps),
        prompt_sha256="seed",
        parent_take_id="",
        seed_text="0",
        keep_frames=int(frames.shape[0]),
        trim_head_frames=0,
        shot_type=1,
        sampled_latent=sampled_lat,
        tail_latent_frames=lat_tail_f or 39,
        raw_trimmed_frames=lat_tail_f if lat_tail_f is not None else int(frames.shape[0]),
        video_rel_path=mp4_rel,
        thumb_rel_path=thumb_rel,
        set_active=False,
    )
    proj = store.load_project()
    proj["shots"].pop("0", None)
    proj["seed_shot"] = {
        "take_id": "shot_0000_seed",
        "frames": int(frames.shape[0]),
        "fps": int(fps),
        "has_tail_latent": bool(meta.get("has_tail_latent", False)),
        "video_rel_path": mp4_rel,
        "thumb_rel_path": thumb_rel,
        "source_path": seed_video_path or "inline_tensor",
    }
    store.save_project(proj)
    return meta
