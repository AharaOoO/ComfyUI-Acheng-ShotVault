"""[N2B] ShotVaultH3Conditioning: Dynamic per-shot H3 multimodal ref & prompt encoder."""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..compat import resample_audio
from ..core.grid_math import (
    AUDIO_VAE_SR,
    FPS_DEFAULT,
    align_frame_count,
    audio_latent_t,
    pad_audio_800,
    video_latent_t,
)

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048


def _adapt_ref_video_canvas(width: int, height: int) -> tuple[int, int]:
    ratio = width / max(1.0, float(height))
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nom_w, nom_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    if width * height < nom_w * nom_h:
        nom_w, nom_h = float(width), float(height)
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def _resize_bhwc(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    samples = image[..., :3].float().movedim(-1, 1)
    try:
        import comfy.utils  # type: ignore

        out = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    except Exception:
        out = F.interpolate(samples, size=(height, width), mode="bilinear", align_corners=False)
    return out.movedim(1, -1).contiguous()


def _encode_ref_audio_tensor(audio_vae, audio: dict[str, Any]) -> tuple[torch.Tensor, int]:
    wav = audio["waveform"].float()
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    sr = int(audio.get("sample_rate", AUDIO_VAE_SR))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", AUDIO_VAE_SR))
    if sr != vae_sr:
        wav = resample_audio(wav, sr, vae_sr)
    wav = pad_audio_800(wav[:1])
    z = audio_vae.encode(wav.movedim(1, -1))
    return z, int(z.shape[-1])


def _empty_h3_av_latent(width: int, height: int, length: int) -> tuple[dict[str, Any], int]:
    import comfy.nested_tensor  # type: ignore

    frame_count = align_frame_count(max(5, int(length)))
    vt = video_latent_t(frame_count)
    at = audio_latent_t(frame_count, FPS_DEFAULT)
    try:
        import comfy.model_management  # type: ignore

        dev = comfy.model_management.intermediate_device()
    except Exception:
        dev = torch.device("cpu")
    v = torch.zeros([1, 24, vt, int(height) // 16, int(width) // 16], device=dev, dtype=torch.float32)
    a = torch.zeros([1, 32, 2, at], device=dev, dtype=torch.float32)
    return {"samples": comfy.nested_tensor.NestedTensor((v, a))}, frame_count


def _attach_minimax_refs(cond: list, ref_blocks: list[dict[str, Any]]) -> list:
    if not ref_blocks:
        return cond
    try:
        import node_helpers  # type: ignore

        return node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
    except Exception:
        out = []
        for emb, meta in cond:
            m = dict(meta or {})
            m["minimax_refs"] = ref_blocks
            out.append([emb, m])
        return out


class ShotVaultH3Conditioning:
    """[N2B] Dynamic per-shot H3 multimodal encoder binding `<Picture i> / <Video k> / <Audio j>` from Director UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "shot_pipe": ("SHOT_PIPE",),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "extra_ref_image_1": ("IMAGE",),
                "extra_ref_image_2": ("IMAGE",),
                "extra_ref_video_1": ("IMAGE",),
                "extra_ref_video_audio_1": ("AUDIO",),
                "extra_ref_audio_1": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "active_ref_map")
    FUNCTION = "encode_shot"
    CATEGORY = "Acheng-ShotVault/Signal"

    def encode_shot(
        self,
        clip,
        shot_pipe: dict[str, Any],
        ref_image_size: str = "match",
        vae=None,
        audio_vae=None,
        extra_ref_image_1: Optional[torch.Tensor] = None,
        extra_ref_image_2: Optional[torch.Tensor] = None,
        extra_ref_video_1: Optional[torch.Tensor] = None,
        extra_ref_video_audio_1: Optional[dict[str, Any]] = None,
        extra_ref_audio_1: Optional[dict[str, Any]] = None,
    ):
        prompt = str(shot_pipe.get("prompt") or "")
        audio_drive_active = bool(shot_pipe.get("audio_drive_active"))
        if audio_drive_active:
            # Audio-MV driven mode uses cond_audio keyframes; mute all <Audio j> reference tokens to avoid conflict.
            prompt = re.sub(r"[ \t]*<Audio\s+\d+\s*>[ \t]*", " ", prompt).strip()
        width = int(shot_pipe.get("width") or 1344)
        height = int(shot_pipe.get("height") or 768)
        length = int(shot_pipe.get("target_frames") or 124)

        latent, frame_count = _empty_h3_av_latent(width, height, length)

        raw_images: list[tuple[torch.Tensor, str]] = []
        for i, img in enumerate(shot_pipe.get("loaded_ref_images") or (), 1):
            if isinstance(img, torch.Tensor) and img.numel() > 0:
                raw_images.append((img, f"shot_img_{i}"))
        for idx, extra_img in enumerate((extra_ref_image_1, extra_ref_image_2), 1):
            if isinstance(extra_img, torch.Tensor) and extra_img.numel() > 0:
                raw_images.append((extra_img, f"canvas_img_{idx}"))
        raw_images = raw_images[:9]

        slot_map = shot_pipe.get("slot_to_ordinal")
        if isinstance(slot_map, dict) and slot_map:
            if any(int(k) != int(v) for k, v in slot_map.items()):
                prompt = re.sub(
                    r"<Picture\s+(\d+)\s*>",
                    lambda m: f"<Picture {slot_map[int(m.group(1))]}>" if int(m.group(1)) in slot_map else m.group(0),
                    prompt,
                )

        raw_videos: list[dict[str, Any]] = []
        for vd in (shot_pipe.get("loaded_ref_videos") or ()):
            if isinstance(vd, dict) and isinstance(vd.get("frames"), torch.Tensor) and vd["frames"].shape[0] >= 5:
                raw_videos.append(vd)
        if isinstance(extra_ref_video_1, torch.Tensor) and extra_ref_video_1.shape[0] >= 5:
            raw_videos.append({"frames": extra_ref_video_1, "audio": extra_ref_video_audio_1, "name": "canvas_video_1"})
        raw_videos = raw_videos[:3]

        raw_audios: list[tuple[dict[str, Any], str]] = []
        if not audio_drive_active:
            for i, aud in enumerate(shot_pipe.get("loaded_ref_audios") or (), 1):
                if isinstance(aud, dict) and isinstance(aud.get("waveform"), torch.Tensor):
                    raw_audios.append((aud, f"shot_audio_{i}"))
            if isinstance(extra_ref_audio_1, dict) and isinstance(extra_ref_audio_1.get("waveform"), torch.Tensor):
                raw_audios.append((extra_ref_audio_1, "canvas_audio_1"))
            raw_audios = raw_audios[:3]

        ref_items: list[dict[str, Any]] = []
        ref_blocks: list[dict[str, Any]] = []
        map_tags: list[str] = []

        for img, label in raw_images:
            h, w = int(img.shape[1]), int(img.shape[2])
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / max(1, w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / max(1, min(w, h)))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize_bhwc(img[:1], tw, th)
            ref_items.append({"type": "image", "data": resized})
            map_tags.append(f"<Picture {len(ref_items)}>={label} ({tw}x{th})")
            if vae is not None:
                z = vae.encode(resized)
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        audio_ord = 0
        vid_ord = 0
        for vd in raw_videos:
            video_frames = vd["frames"]
            soundtrack = None if audio_drive_active else vd.get("audio")
            vh, vw = int(video_frames.shape[1]), int(video_frames.shape[2])
            cw, ch = _adapt_ref_video_canvas(vw, vh)
            frames = _resize_bhwc(video_frames, cw, ch)
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = int(frames.shape[0])
            while n >= 5 and n % 17 != 5:
                n -= 1
            if n < 5:
                continue
            frames = frames[:n]

            has_vid_audio = (not audio_drive_active) and isinstance(soundtrack, dict) and isinstance(soundtrack.get("waveform"), torch.Tensor)
            if has_vid_audio:
                ref_items.append({"type": "audio"})
                audio_ord += 1
                map_tags.append(f"<Audio {audio_ord}>={vd.get('name', 'video')}(soundtrack)")

            sample_idx = list(range(0, frames.shape[0], FPS_DEFAULT // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            })
            vid_ord += 1
            map_tags.append(f"<Video {vid_ord}>={vd.get('name', 'video')} ({n}f)")

            if vae is not None:
                z = vae.encode(frames)
                audio_latent, ref_audio_t = (None, 0)
                if has_vid_audio and audio_vae is not None:
                    audio_latent, ref_audio_t = _encode_ref_audio_tensor(audio_vae, soundtrack)
                ref_blocks.append({
                    "kind": "video_audio" if ref_audio_t else "video",
                    "latent_t": int(z.shape[2]),
                    "latent_h": ch // 16,
                    "latent_w": cw // 16,
                    "ref_audio_t": ref_audio_t,
                    "latent": z,
                    "audio_latent": audio_latent,
                })

        for aud, label in raw_audios:
            ref_items.append({"type": "audio"})
            audio_ord += 1
            map_tags.append(f"<Audio {audio_ord}>={label}")
            if audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio_tensor(audio_vae, aud)
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        if ref_items:
            try:
                tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            except TypeError:
                tokens = clip.tokenize(prompt)
        else:
            tokens = clip.tokenize(prompt)

        cond = clip.encode_from_tokens_scheduled(tokens)
        cond = _attach_minimax_refs(cond, ref_blocks)
        active_ref_map = shot_pipe.get("active_ref_summary") or ", ".join(map_tags)
        if map_tags:
            active_ref_map = " | ".join(map_tags)
        return (cond, latent, active_ref_map)
