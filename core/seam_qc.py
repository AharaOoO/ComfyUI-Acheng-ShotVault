"""FFmpeg MP4 / ProRes writer, lightweight JPEG thumbnailer, and 3-second seam QC preview builder."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch

from ..compat import resample_audio, resolve_ffmpeg_exe
from .grid_math import DELIVERY_AUDIO_SR, FPS_DEFAULT


def write_jpg_thumb(frame_hwc: torch.Tensor, out_path: str, max_width: int = 480) -> str:
    """Write a lightweight JPEG thumbnail from `[H, W, 3]` (0..1) frame tensor."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img = frame_hwc[..., :3].detach().cpu().float().clamp(0.0, 1.0)
    h, w = int(img.shape[0]), int(img.shape[1])
    if w > max_width and max_width > 0:
        new_h = max(2, round(h * max_width / float(w)))
        chw = torch.nn.functional.interpolate(
            img.permute(2, 0, 1).unsqueeze(0), size=(new_h, max_width), mode="bilinear", align_corners=False
        )
        img = chw.squeeze(0).permute(1, 2, 0)
    u8 = (img * 255.0).round().to(torch.uint8).numpy()
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(u8).save(out_path, format="JPEG", quality=85)
        return out_path
    except Exception:
        import av  # type: ignore

        with av.open(out_path, mode="w") as container:
            stream = container.add_stream("mjpeg", rate=1)
            stream.width = int(u8.shape[1])
            stream.height = int(u8.shape[0])
            stream.pix_fmt = "yuvj420p"
            frame = av.VideoFrame.from_ndarray(u8, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
        return out_path


def _normalize_stereo_pcm(audio_wav: Optional[torch.Tensor], audio_sr: int, frame_count: int, fps: int) -> np.ndarray:
    """Return contiguous `<f4` interleaved stereo array `[samples, 2]` at 48kHz."""
    target_samples = max(1, round(int(frame_count) / float(fps) * DELIVERY_AUDIO_SR))
    if audio_wav is None or audio_wav.numel() == 0:
        return np.zeros((target_samples, 2), dtype="<f4")
    wav = audio_wav.detach().cpu().float()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    wav_48k = resample_audio(wav, int(audio_sr), DELIVERY_AUDIO_SR)
    if wav_48k.shape[1] < target_samples:
        wav_48k = torch.nn.functional.pad(wav_48k, (0, target_samples - wav_48k.shape[1]))
    else:
        wav_48k = wav_48k[:, :target_samples]
    return np.ascontiguousarray(wav_48k.transpose(0, 1).numpy(), dtype="<f4")


def write_mp4_with_pcm(
    frames_hwc: torch.Tensor,
    audio_wav: Optional[torch.Tensor],
    audio_sr: int,
    fps: int,
    out_path: str,
    crf: int = 18,
    codec: str = "h264",
) -> str:
    """Stream `[F, H, W, 3]` video in 16-frame chunks + stereo 48kHz PCM via a single FFmpeg pass."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ffmpeg_exe = resolve_ffmpeg_exe()
    n_frames, height, width = int(frames_hwc.shape[0]), int(frames_hwc.shape[1]), int(frames_hwc.shape[2])
    pcm_interleaved = _normalize_stereo_pcm(audio_wav, int(audio_sr), n_frames, int(fps))

    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tf:
        pcm_tmp = tf.name
        tf.write(pcm_interleaved.tobytes())

    try:
        cmd = [
            ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(int(fps)), "-i", "pipe:0",
            "-f", "f32le", "-ar", str(DELIVERY_AUDIO_SR), "-ac", "2", "-i", pcm_tmp,
        ]
        if codec == "prores_ks":
            cmd += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s16le"]
        else:
            cmd += [
                "-c:v", "libx264", "-crf", str(int(crf)), "-preset", "fast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart",
            ]
        cmd += ["-shortest", out_path]
        with tempfile.TemporaryFile() as err_f:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err_f)
            assert proc.stdin is not None
            try:
                for start in range(0, n_frames, 16):
                    chunk_u8 = (
                        frames_hwc[start:start + 16, ..., :3].detach().cpu().float().clamp(0.0, 1.0) * 255.0
                    ).round().to(torch.uint8).contiguous().numpy()
                    proc.stdin.write(memoryview(chunk_u8))
            except BrokenPipeError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            ret = proc.wait()
            if ret != 0:
                err_f.seek(0)
                err_txt = err_f.read().decode("utf-8", "replace")
                raise RuntimeError(f"ShotVault FFmpeg encode failed ({ret}): {err_txt}")
        return out_path
    finally:
        if os.path.exists(pcm_tmp):
            try:
                os.remove(pcm_tmp)
            except OSError:
                pass


def build_seam_qc_clip(
    prev_clean_tail: torch.Tensor,
    prev_audio_48k: Optional[torch.Tensor],
    curr_frames: torch.Tensor,
    curr_audio_48k: Optional[torch.Tensor],
    fps: int,
    out_path: str,
) -> Optional[str]:
    """Create 3-second (`prev 1s + curr 2s`) 48kHz audio+video seam inspection preview."""
    if prev_clean_tail is None or prev_clean_tail.shape[0] == 0 or curr_frames.shape[0] == 0:
        return None
    n_prev = min(int(prev_clean_tail.shape[0]), int(fps))
    n_curr = min(int(curr_frames.shape[0]), 2 * int(fps))
    h, w = int(curr_frames.shape[1]), int(curr_frames.shape[2])
    p_frames = prev_clean_tail[-n_prev:, ..., :3].detach().cpu().float()
    if p_frames.shape[1] != h or p_frames.shape[2] != w:
        p_frames = torch.nn.functional.interpolate(
            p_frames.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False
        ).movedim(1, -1)
    combined_frames = torch.cat([p_frames, curr_frames[:n_curr, ..., :3].detach().cpu().float()], dim=0)

    def _tail_audio(wav: Optional[torch.Tensor], n_f: int, from_end: bool) -> torch.Tensor:
        target = max(1, round(n_f / float(fps) * DELIVERY_AUDIO_SR))
        if wav is None or wav.numel() == 0:
            return torch.zeros((2, target), dtype=torch.float32)
        w = wav.detach().cpu().float()
        if w.ndim == 3:
            w = w[0]
        if w.shape[0] == 1:
            w = w.repeat(2, 1)
        elif w.shape[0] > 2:
            w = w[:2]
        sl = w[:, -target:] if from_end else w[:, :target]
        if sl.shape[1] < target:
            sl = torch.nn.functional.pad(sl, (0, target - sl.shape[1]))
        return sl[:, :target]

    comb_audio = torch.cat(
        [_tail_audio(prev_audio_48k, n_prev, True), _tail_audio(curr_audio_48k, n_curr, False)],
        dim=1,
    )
    return write_mp4_with_pcm(combined_frames, comb_audio, DELIVERY_AUDIO_SR, fps, out_path, crf=21)


def analyze_seam_diagnostics(
    prev_clean_tail: Optional[torch.Tensor],
    curr_frames: torch.Tensor,
    fps: int = FPS_DEFAULT,
) -> dict:
    """SGUN-inspired seam pixel-jump and brightness-step diagnostics at `join_frame`."""
    if prev_clean_tail is None or prev_clean_tail.shape[0] == 0 or curr_frames.shape[0] == 0:
        return {"status": "ok", "join_frame": 0, "join_sec": 0.0, "fps": int(fps), "messages": ["首段或硬切起始无前段接缝。"], "events": []}
    fps_i = max(1, int(fps))
    n_prev = min(int(prev_clean_tail.shape[0]), fps_i)
    n_curr = min(int(curr_frames.shape[0]), 2 * fps_i)

    def _down(t: torch.Tensor) -> np.ndarray:
        x = t[..., :3].detach().cpu().float().clamp(0.0, 1.0).permute(0, 3, 1, 2)
        small = torch.nn.functional.interpolate(x, size=(54, 96), mode="area")
        return small.permute(0, 2, 3, 1).numpy()

    seq = np.concatenate([_down(prev_clean_tail[-n_prev:]), _down(curr_frames[:n_curr])], axis=0)
    diffs = [0.0]
    shifts = [0.0]
    means = seq.mean(axis=(1, 2))
    for i in range(1, seq.shape[0]):
        diffs.append(float(np.abs(seq[i] - seq[i - 1]).mean()))
        shifts.append(float(np.max(np.abs(means[i] - means[i - 1]))))

    seam_events = []
    interior_events = []
    seam_win_end = min(len(diffs), n_prev + 6)
    for idx in range(n_prev, len(diffs)):
        nb = [j for j in range(max(1, idx - 4), min(len(diffs), idx + 5)) if j != idx]
        base_d = float(np.median([diffs[j] for j in nb])) if nb else 0.0
        base_s = float(np.median([shifts[j] for j in nb])) if nb else 0.0
        kind = None
        if shifts[idx] > max(0.08, base_s * 4.0):
            kind = "brightness_step"
        elif diffs[idx] > max(0.16, base_d * 4.0):
            kind = "pixel_jump"
        if kind:
            ev = {
                "frame": idx,
                "offset_frames": int(idx - n_prev),
                "kind": kind,
                "difference": round(diffs[idx], 4),
                "rgb_shift": round(shifts[idx], 4),
            }
            if idx < seam_win_end:
                seam_events.append(ev)
            else:
                interior_events.append(ev)

    messages_zh: list[str] = []
    messages_en: list[str] = []
    for kind, label_zh, label_en in (
        ("pixel_jump", "接缝处疑似画面跳变", "possible pixel jump at seam"),
        ("brightness_step", "接缝处疑似亮度/颜色台阶", "possible brightness/color step at seam"),
    ):
        matching = [e for e in seam_events if e["kind"] == kind]
        if matching:
            sec = (matching[0]["frame"] - n_prev) / float(fps_i)
            messages_zh.append(f"接缝 +{sec:.2f}s 处{label_zh}。")
            messages_en.append(f"Seam +{sec:.2f}s: {label_en}.")
    if seam_events:
        messages_zh.append("请切换原版/修复版预览，可一键单独修复本处接缝。")
        messages_en.append("Compare Original vs Repaired preview, or repair this seam directly.")
    elif interior_events:
        first_in = interior_events[0]
        sec_in = (first_in["frame"] - n_prev) / float(fps_i)
        messages_zh.append(f"接缝平滑；段内 +{sec_in:.2f}s 存在光影/动作变化（非接缝问题）。")
        messages_en.append(f"Seam smooth; interior flash/motion at +{sec_in:.2f}s (not a seam defect).")

    return {
        "status": "warning" if seam_events else "ok",
        "join_frame": int(n_prev),
        "join_sec": round(n_prev / float(fps_i), 2),
        "fps": fps_i,
        "messages": messages_zh,
        "messages_zh": messages_zh,
        "messages_en": messages_en,
        "events": seam_events,
        "interior_events": interior_events,
    }
