"""Turnaround sheet splitter (white, grey studio, or hairline-divided 1x4 & 2x2) and dynamic `<Picture N>` asset loader."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..compat import resample_audio
from .grid_math import AUDIO_VAE_SR, pad_audio_800


def _trim_uniform_borders(panel: torch.Tensor, tol: float = 0.045, margin: int = 4) -> torch.Tensor:
    """Trim uniform studio/white outer margins detected from corner background color."""
    if panel.numel() == 0:
        return panel
    h, w = int(panel.shape[0]), int(panel.shape[1])
    if h < 12 or w < 12:
        return panel
    corners = torch.cat([
        panel[:4, :4, :3].reshape(-1, 3),
        panel[:4, -4:, :3].reshape(-1, 3),
        panel[-4:, :4, :3].reshape(-1, 3),
        panel[-4:, -4:, :3].reshape(-1, 3),
    ], dim=0)
    bg = corners.median(dim=0).values
    fg = (panel[..., :3] - bg).abs().max(dim=-1).values > tol
    ys, xs = torch.where(fg)
    if ys.numel() < 16 or xs.numel() < 16:
        return panel
    y0 = max(0, int(ys.min().item()) - margin)
    y1 = min(h, int(ys.max().item()) + 1 + margin)
    x0 = max(0, int(xs.min().item()) - margin)
    x1 = min(w, int(xs.max().item()) + 1 + margin)
    return panel[y0:y1, x0:x1].contiguous()


def split_white_turnaround(
    image: torch.Tensor,
    layout: str = "auto",
    white_thresh: float = 0.94,
) -> list[torch.Tensor]:
    """Split a studio/white/divider 1x4 or 2x2 turnaround sheet into clean individual view panels."""
    img = image[0] if image.ndim == 4 else image
    img = img[..., :3].float().contiguous()
    h, w = int(img.shape[0]), int(img.shape[1])
    col_std = img.std(dim=0).mean(dim=-1)
    row_std = img.std(dim=1).mean(dim=-1)

    if layout == "auto":
        mid_r0, mid_r1 = int(0.42 * h), max(int(0.42 * h) + 1, int(0.58 * h))
        mid_c0, mid_c1 = int(0.42 * w), max(int(0.42 * w) + 1, int(0.58 * w))
        mid_row_min = float(row_std[mid_r0:mid_r1].min().item())
        mid_col_min = float(col_std[mid_c0:mid_c1].min().item())
        if w < h * 1.35 and mid_row_min < 0.035 and mid_col_min < 0.035:
            layout = "2x2"
        else:
            layout = "1x4" if w >= h * 1.2 else "2x2"

    panels: list[torch.Tensor] = []
    if layout == "1x4":
        search_bands = [(0.18, 0.36), (0.44, 0.62), (0.66, 0.84)]
        cuts = []
        for lo_r, hi_r in search_bands:
            s = max(1, int(lo_r * w))
            e = min(w - 1, max(s + 1, int(hi_r * w)))
            if cuts:
                s = max(s, cuts[-1] + max(16, int(0.10 * w)))
                if s >= e:
                    e = min(w - 1, s + 8)
            best_x = s + int(torch.argmin(col_std[s:e]).item())
            cuts.append(best_x)
        shave = 3 if w >= 256 else 1
        bounds = [
            (0, max(1, cuts[0] - shave)),
            (min(w - 1, cuts[0] + shave), max(cuts[0] + shave + 1, cuts[1] - shave)),
            (min(w - 1, cuts[1] + shave), max(cuts[1] + shave + 1, cuts[2] - shave)),
            (min(w - 1, cuts[2] + shave), w),
        ]
        for x0, x1 in bounds:
            panels.append(_trim_uniform_borders(img[:, x0:x1, :]))
    else:
        r_mid = int(0.40 * h) + int(torch.argmin(row_std[int(0.40 * h):max(int(0.40 * h) + 1, int(0.60 * h))]).item())
        c_mid = int(0.40 * w) + int(torch.argmin(col_std[int(0.40 * w):max(int(0.40 * w) + 1, int(0.60 * w))]).item())
        shave = 2 if min(h, w) >= 256 else 0
        for y0, y1 in ((0, max(1, r_mid - shave)), (min(h - 1, r_mid + shave), h)):
            for x0, x1 in ((0, max(1, c_mid - shave)), (min(w - 1, c_mid + shave), w)):
                panels.append(_trim_uniform_borders(img[y0:y1, x0:x1, :]))
    return panels


split_turnaround_sheet = split_white_turnaround


def select_primary_character_views(panels: list[torch.Tensor]) -> list[torch.Tensor]:
    """Select portrait face closeup + primary front full-body view as batched `[1, H, W, 3]` tensors."""
    if not panels:
        return []
    scored = []
    for idx, p in enumerate(panels):
        ph, pw = int(p.shape[0]), int(p.shape[1])
        bg = p[:4, :4, :3].reshape(-1, 3).median(dim=0).values
        bottom_band = p[max(0, int(ph * 0.88)):, :, :3]
        bottom_occ = float(((bottom_band - bg).abs().max(dim=-1).values > 0.05).float().mean().item())
        scored.append((idx, pw, bottom_occ, p.unsqueeze(0).contiguous()))
    portrait = max(scored, key=lambda item: (item[2], item[1]))
    others = [item for item in scored if item[0] != portrait[0]]
    ordered = [portrait[3]]
    if others:
        full_body = max(others, key=lambda item: item[1])
        ordered.append(full_body[3])
        for item in scored:
            if item[0] not in (portrait[0], full_body[0]):
                ordered.append(item[3])
    return ordered


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_AUD_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def scan_assets_pool(assets_root_dir: str) -> dict[str, dict[str, str]]:
    """Scan `assets/{chars,props,scenes,voices}/` into stem -> path maps."""
    pool: dict[str, dict[str, str]] = {"images": {}, "voices": {}}
    if not assets_root_dir or not os.path.isdir(assets_root_dir):
        return pool
    for sub in ("chars", "props", "scenes"):
        d = os.path.join(assets_root_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(_IMG_EXTS):
                stem = os.path.splitext(fn)[0]
                pool["images"][stem] = os.path.join(d, fn)
    vd = os.path.join(assets_root_dir, "voices")
    if os.path.isdir(vd):
        for fn in sorted(os.listdir(vd)):
            if fn.lower().endswith(_AUD_EXTS):
                stem = os.path.splitext(fn)[0]
                pool["voices"][stem] = os.path.join(vd, fn)
    return pool


def resolve_shot_assets(
    assets_pool: dict[str, dict[str, str]],
    prompt_text: str,
    auto_split_turnaround: bool = False,
) -> dict[str, Any]:
    """Dynamically bind referenced assets in a shot and compact `<Picture 1..N>` indices."""
    matched_imgs = []
    for name, path in (assets_pool.get("images") or {}).items():
        if name and name in prompt_text:
            matched_imgs.append({"name": name, "path": path})
    matched_voices = []
    for name, path in (assets_pool.get("voices") or {}).items():
        if name and name in prompt_text:
            matched_voices.append({"name": name, "path": path})
    reindexed_hint = ", ".join(f"<Picture {i+1}>={item['name']}" for i, item in enumerate(matched_imgs))
    return {
        "image_assets": matched_imgs,
        "voice_assets": matched_voices,
        "turnaround_split_enabled": bool(auto_split_turnaround),
        "turnaround_hint": (
            "保持多角度四视图拆图人物面部与服饰一致，禁止跨视角串脸。"
            if auto_split_turnaround and matched_imgs
            else ""
        ),
        "reindexed_hint": reindexed_hint,
    }


def _load_image_tensor(path: str) -> torch.Tensor:
    from PIL import Image  # type: ignore

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _load_audio_dict(path: str) -> dict[str, Any] | None:
    """Decode an audio file into a ComfyUI AUDIO dict `{'waveform': [1, C, S], 'sample_rate': sr}`."""
    if not path or not os.path.isfile(path):
        return None
    try:
        import av  # type: ignore

        chunks = []
        sr = 48000
        with av.open(path) as container:
            a_stream = next((s for s in container.streams if s.type == "audio"), None)
            if a_stream is None:
                return None
            sr = int(a_stream.sample_rate or 48000)
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
        if chunks:
            return {"waveform": torch.cat(chunks, dim=1).unsqueeze(0).contiguous(), "sample_rate": sr}
    except Exception:
        return None
    return None


def _load_video_frames_and_audio(path: str, target_fps: int = 24, max_frames: int = 362) -> dict[str, Any] | None:
    """Decode RGB frames `[F, H, W, 3]` (0..1) and optional soundtrack AUDIO dict from video."""
    if not path or not os.path.isfile(path):
        return None
    try:
        import av  # type: ignore

        frames_rgb: list[np.ndarray] = []
        audio_chunks: list[torch.Tensor] = []
        src_fps = float(target_fps)
        src_sr = 48000
        with av.open(path) as container:
            v_stream = next((s for s in container.streams if s.type == "video"), None)
            if v_stream is None:
                return None
            if v_stream.average_rate:
                src_fps = float(v_stream.average_rate)
            raw_limit = max(int(max_frames), round(int(max_frames) * src_fps / max(1.0, float(target_fps))) + 4)
            for frame in container.decode(video=0):
                frames_rgb.append(frame.to_ndarray(format="rgb24"))
                if len(frames_rgb) >= raw_limit:
                    break
        if not frames_rgb:
            return None
        # Audio MUST open a fresh demuxer handle; decoding video=0 to raw_limit/EOF
        # advances the shared container cursor past interleaved audio packets.
        with av.open(path) as a_container:
            a_stream = next((s for s in a_container.streams if s.type == "audio"), None)
            if a_stream is not None:
                src_sr = int(a_stream.sample_rate or 48000)
                max_samples = round(len(frames_rgb) / max(1.0, src_fps) * src_sr)
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
                    if max_samples > 0 and total_samples + t.shape[1] > max_samples:
                        t = t[:, : max(0, max_samples - total_samples)]
                    if t.shape[1] > 0:
                        audio_chunks.append(t)
                        total_samples += int(t.shape[1])
                    if max_samples > 0 and total_samples >= max_samples:
                        break
        vid_t = torch.from_numpy(np.stack(frames_rgb, axis=0)).float().div_(255.0)
        if abs(src_fps - float(target_fps)) > 0.25 and vid_t.shape[0] > 1:
            new_n = max(1, round(vid_t.shape[0] * float(target_fps) / src_fps))
            idx = torch.linspace(0, vid_t.shape[0] - 1, steps=new_n).round().long()
            vid_t = vid_t[idx]
        if vid_t.shape[0] > max_frames:
            vid_t = vid_t[:max_frames]
        audio_dict = None
        if audio_chunks:
            audio_dict = {"waveform": torch.cat(audio_chunks, dim=1).unsqueeze(0).contiguous(), "sample_rate": src_sr}
        return {"frames": vid_t.contiguous(), "audio": audio_dict, "name": os.path.basename(path)}
    except Exception:
        return None


def resolve_media_file(raw_path: str, input_dir: str = "", project_root: str = "") -> str:
    """Resolve a user-uploaded relative/input/project media path without traversing outside allowed roots."""
    p = (raw_path or "").strip()
    if not p:
        return ""
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    clean = p.replace("\\", "/")
    if clean.startswith("input/"):
        clean = clean[6:]
    for base in (input_dir, project_root, os.path.dirname(input_dir) if input_dir else ""):
        if not base:
            continue
        base_abs = os.path.abspath(base)
        cand = os.path.abspath(os.path.join(base_abs, clean))
        if cand.startswith(base_abs + os.sep) and os.path.isfile(cand):
            return cand
    return ""


def load_and_prepare_shot_assets(bound_assets: dict[str, Any]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Load real IMAGE tensors (applying turnaround splitting when enabled) and matched AUDIO assets."""
    images_out: list[torch.Tensor] = []
    auto_split = bool(bound_assets.get("turnaround_split_enabled", False))
    for item in (bound_assets.get("image_assets") or ()):
        path = item.get("path") or ""
        if not path or not os.path.isfile(path):
            continue
        tensor = _load_image_tensor(path)
        name_low = os.path.basename(path).lower()
        is_turnaround_name = any(k in name_low for k in ("4view", "四视图", "turnaround", "三视图", "view"))
        _, h, w, _ = tensor.shape
        if auto_split and (is_turnaround_name or w >= h * 1.35):
            views = select_primary_character_views(split_white_turnaround(tensor, layout="auto"))
            images_out.extend(views[:2] if len(views) >= 2 else views)
        else:
            images_out.append(tensor)

    voices_out: list[dict[str, Any]] = []
    for item in (bound_assets.get("voice_assets") or ()):
        audio_d = _load_audio_dict(item.get("path") or "")
        if audio_d is not None:
            voices_out.append(audio_d)
    return images_out, voices_out


def load_shot_multimodal_refs(
    shot: dict[str, Any],
    global_cfg: dict[str, Any] | None = None,
    input_dir: str = "",
    project_root: str = "",
    auto_split_turnaround: bool = False,
    fps: int = 24,
    max_frames: int = 362,
    skip_ref_audios: bool = False,
) -> dict[str, Any]:
    """Merge global + per-shot + asset-pool references into ordered images/videos/audios tensors & `<Tag>` map."""
    gcfg = global_cfg or {}
    inherit = bool(shot.get("inherit_global_refs", True))
    bound = shot.get("assets") or {}

    slot_paths: list[str] = [""] * 9
    used_paths: set[str] = set()
    has_slots = ("ref_slots" in shot) or ("global_ref_slots" in gcfg)
    if has_slots:
        own_slots = [str(x or "").strip() for x in (shot.get("ref_slots") or ())][:9]
        glob_slots = [str(x or "").strip() for x in (gcfg.get("global_ref_slots") or gcfg.get("global_ref_images") or ())][:9]
        for k in range(9):
            own_k = own_slots[k] if k < len(own_slots) else ""
            glob_k = glob_slots[k] if (inherit and k < len(glob_slots)) else ""
            raw_cand = own_k or glob_k
            if raw_cand:
                rp = resolve_media_file(raw_cand, input_dir, project_root)
                if rp:
                    slot_paths[k] = rp
                    used_paths.add(rp)
        if not any(own_slots):
            for p_fallback in (shot.get("ref_images") or ()):
                rp = resolve_media_file(str(p_fallback or "").strip(), input_dir, project_root)
                if rp and rp not in used_paths:
                    for k in range(9):
                        if not slot_paths[k]:
                            slot_paths[k] = rp
                            used_paths.add(rp)
                            break
    else:
        raw_img_list = (list(gcfg.get("global_ref_images") or ()) if inherit else []) + list(shot.get("ref_images") or ())
        for raw_p in raw_img_list:
            rp = resolve_media_file(str(raw_p), input_dir, project_root)
            if rp and rp not in used_paths:
                for k in range(9):
                    if not slot_paths[k]:
                        slot_paths[k] = rp
                        used_paths.add(rp)
                        break

    for item in (bound.get("image_assets") or ()):
        rp = str(item.get("path") or "")
        if rp and os.path.isfile(rp) and rp not in used_paths:
            for k in range(9):
                if not slot_paths[k]:
                    slot_paths[k] = rp
                    used_paths.add(rp)
                    break

    images_out: list[torch.Tensor] = []
    img_names: list[str] = []
    slot_images: list[Optional[torch.Tensor]] = [None] * 9
    slot_to_ordinal: dict[int, int] = {}
    for k, rp in enumerate(slot_paths):
        if not rp:
            continue
        tensor = _load_image_tensor(rp)
        name_low = os.path.basename(rp).lower()
        is_turnaround = any(kw in name_low for kw in ("4view", "四视图", "turnaround", "三视图", "view"))
        _, h, w, _ = tensor.shape
        if auto_split_turnaround and (is_turnaround or w >= h * 1.35):
            views = select_primary_character_views(split_white_turnaround(tensor, layout="auto"))
            picks = views[:2] if len(views) >= 2 else views
            if picks:
                slot_images[k] = picks[0]
            for v_idx, v in enumerate(picks):
                if len(images_out) < 9:
                    images_out.append(v)
                    img_names.append(os.path.basename(rp))
                    if v_idx == 0:
                        slot_to_ordinal[k + 1] = len(images_out)
        else:
            slot_images[k] = tensor
            if len(images_out) < 9:
                images_out.append(tensor)
                img_names.append(os.path.basename(rp))
                slot_to_ordinal[k + 1] = len(images_out)

    videos_out: list[dict[str, Any]] = []
    for entry in list(shot.get("ref_videos") or ())[:3]:
        if isinstance(entry, dict):
            raw_vp = str(entry.get("path") or "")
            with_aud = bool(entry.get("with_audio", True))
        else:
            raw_vp = str(entry or "")
            with_aud = True
        rp = resolve_media_file(raw_vp, input_dir, project_root)
        if not rp:
            continue
        loaded = _load_video_frames_and_audio(rp, target_fps=fps, max_frames=max_frames)
        if loaded is not None:
            if not with_aud or skip_ref_audios:
                loaded["audio"] = None
            videos_out.append(loaded)

    audios_out: list[dict[str, Any]] = []
    aud_names: list[str] = []
    if not skip_ref_audios:
        aud_paths: list[str] = []
        seen_aud: set[str] = set()
        raw_aud_list = (list(gcfg.get("global_ref_audios") or ()) if inherit else []) + list(shot.get("ref_audios") or ())
        for raw_ap in raw_aud_list:
            rp = resolve_media_file(str(raw_ap), input_dir, project_root)
            if rp and rp not in seen_aud:
                seen_aud.add(rp)
                aud_paths.append(rp)
        for item in (bound.get("voice_assets") or ()):
            rp = str(item.get("path") or "")
            if rp and os.path.isfile(rp) and rp not in seen_aud:
                seen_aud.add(rp)
                aud_paths.append(rp)

        for rp in aud_paths[:3]:
            ad = _load_audio_dict(rp)
            if ad is not None:
                audios_out.append(ad)
                aud_names.append(os.path.basename(rp))

    tags: list[str] = [f"<Picture {i+1}>={nm}" for i, nm in enumerate(img_names)]
    aud_ord = 0
    for k, vd in enumerate(videos_out, 1):
        if vd.get("audio") is not None:
            aud_ord += 1
            tags.append(f"<Audio {aud_ord}>={vd['name']}(audio)")
        tags.append(f"<Video {k}>={vd['name']}")
    for nm in aud_names:
        aud_ord += 1
        tags.append(f"<Audio {aud_ord}>={nm}")

    return {
        "images": images_out,
        "slot_images": slot_images,
        "slot_to_ordinal": slot_to_ordinal,
        "videos": videos_out,
        "audios": audios_out,
        "summary": ", ".join(tags),
    }


def _norm_stereo_2d(wav: torch.Tensor) -> torch.Tensor:
    w = wav.detach().cpu().float()
    if w.ndim == 3:
        w = w[0]
    elif w.ndim == 1:
        w = w.unsqueeze(0)
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    elif w.shape[0] > 2:
        w = w[:2]
    return w.contiguous()


def slice_1114_prev_audio_guide(
    raw_tail_32k: torch.Tensor,
    overlap_frames: int,
    audio_guide_frames: int = 0,
    fps: int = 24,
) -> dict[str, Any]:
    """Slice 1114-style continuation audio guide `[-ov_frames : -(ov_frames - ag_frames)]` @ 32kHz,
    dropping the final 2 pop-prone boundary frames (`22f->20f`, `39f->37f`, `56f->54f`) without tail zero-cliffs.
    """
    fps_f = float(max(1, int(fps)))
    ov = max(1, int(overlap_frames))
    ag = int(audio_guide_frames) if int(audio_guide_frames) > 0 else max(1, ov - 2)
    ag = max(1, min(ov, ag))

    ov_samples = max(1, round(ov / fps_f * AUDIO_VAE_SR))
    ag_samples = max(1, min(ov_samples, round(ag / fps_f * AUDIO_VAE_SR)))
    guard_samples = max(0, ov_samples - ag_samples)

    tail = _norm_stereo_2d(raw_tail_32k)
    if tail.shape[1] < ov_samples:
        tail = F.pad(tail, (ov_samples - int(tail.shape[1]), 0))
    window = tail[:, -ov_samples:]
    a_slice = window[:, : ov_samples - guard_samples] if guard_samples > 0 else window
    return {"waveform": a_slice.unsqueeze(0).contiguous(), "sample_rate": AUDIO_VAE_SR}


def build_continuation_driving_audio(
    shot_audio_ref: Optional[dict[str, Any]],
    *,
    curr_idx: int,
    sched: dict[str, Any],
    fps: int,
    av_sync_lead_sec: float = 0.0,
    parent_ctx: Optional[dict[str, Any]] = None,
    per_shot_audio_map: Optional[dict[int, dict[str, Any]]] = None,
    seed_ctx: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Build 32kHz `shot_audio_guide` (`[overlap_tail_audio + curr_audio + optional_pad_lookahead]`) and net delivery audio."""
    empty_guide = {"waveform": torch.zeros((1, 2, 800), dtype=torch.float32), "sample_rate": AUDIO_VAE_SR}
    if not isinstance(shot_audio_ref, dict) or not isinstance(shot_audio_ref.get("waveform"), torch.Tensor) or shot_audio_ref["waveform"].numel() == 0:
        return empty_guide, None

    raw_w = _norm_stereo_2d(shot_audio_ref["waveform"])
    raw_sr = int(shot_audio_ref.get("sample_rate", 48000))
    delivery_ref_audio = {"waveform": raw_w.unsqueeze(0).contiguous(), "sample_rate": raw_sr}

    if shot_audio_ref.get("guide_prealigned") and isinstance(shot_audio_ref.get("guide_waveform"), torch.Tensor):
        gw_sr = int(shot_audio_ref.get("guide_sample_rate", raw_sr))
        gw_32k = resample_audio(_norm_stereo_2d(shot_audio_ref["guide_waveform"]), gw_sr, AUDIO_VAE_SR)
        lead_sec = float(av_sync_lead_sec)
        if lead_sec > 0:
            cut_s = round(lead_sec * AUDIO_VAE_SR)
            gw_32k = gw_32k[:, cut_s:] if cut_s < gw_32k.shape[1] else torch.zeros((2, 800), dtype=torch.float32)
        elif lead_sec < 0:
            gw_32k = torch.cat([torch.zeros((2, round((-lead_sec) * AUDIO_VAE_SR)), dtype=torch.float32), gw_32k], dim=1)
        return {"waveform": pad_audio_800(gw_32k.unsqueeze(0)).contiguous(), "sample_rate": AUDIO_VAE_SR}, delivery_ref_audio

    w_32k = resample_audio(raw_w, raw_sr, AUDIO_VAE_SR)
    trim_sec = int(sched.get("trim_head_frames", 0)) / float(max(1, int(fps)))
    shift_sec = trim_sec - float(av_sync_lead_sec)
    is_cont = bool(sched.get("is_continuation", False))

    if shift_sec > 0:
        need_pre = max(1, round(shift_sec * AUDIO_VAE_SR))
        pre_src: Optional[torch.Tensor] = None
        if is_cont and isinstance(parent_ctx, dict):
            pt = (parent_ctx.get("tensors") or {}).get("audio_32k_tail")
            if isinstance(pt, torch.Tensor) and pt.numel() > 0 and float(pt.abs().max().item()) > 1e-6:
                pre_src = _norm_stereo_2d(pt)
        if pre_src is None and is_cont and per_shot_audio_map and (curr_idx - 1) in per_shot_audio_map:
            prev_ref = per_shot_audio_map[curr_idx - 1]
            if isinstance(prev_ref, dict) and isinstance(prev_ref.get("waveform"), torch.Tensor) and prev_ref["waveform"].numel() > 0:
                pre_src = resample_audio(_norm_stereo_2d(prev_ref["waveform"]), int(prev_ref.get("sample_rate", 48000)), AUDIO_VAE_SR)
        if pre_src is None and is_cont and curr_idx == 1 and isinstance(seed_ctx, dict):
            st = (seed_ctx.get("tensors") or {}).get("audio_32k_tail")
            if isinstance(st, torch.Tensor) and st.numel() > 0 and float(st.abs().max().item()) > 1e-6:
                pre_src = _norm_stereo_2d(st)

        if pre_src is not None and pre_src.shape[1] > 0:
            if pre_src.shape[1] >= need_pre:
                pre_32k = pre_src[:, -need_pre:]
            else:
                pre_32k = F.pad(pre_src, (need_pre - int(pre_src.shape[1]), 0))
        else:
            pre_32k = torch.zeros((2, need_pre), dtype=torch.float32)
        w_32k = torch.cat([pre_32k, w_32k], dim=1)
    elif shift_sec < 0:
        cut_s = round((-shift_sec) * AUDIO_VAE_SR)
        w_32k = w_32k[:, cut_s:] if cut_s < w_32k.shape[1] else torch.zeros((2, 800), dtype=torch.float32)

    target_s = round(int(sched.get("target_frames", 0)) / float(max(1, int(fps))) * AUDIO_VAE_SR)
    if target_s > int(w_32k.shape[1]) and per_shot_audio_map and (curr_idx + 1) in per_shot_audio_map:
        nxt = per_shot_audio_map[curr_idx + 1]
        if isinstance(nxt, dict) and isinstance(nxt.get("waveform"), torch.Tensor) and nxt["waveform"].numel() > 0:
            nxt_32k = resample_audio(_norm_stereo_2d(nxt["waveform"]), int(nxt.get("sample_rate", 48000)), AUDIO_VAE_SR)
            need_tail = target_s - int(w_32k.shape[1])
            w_32k = torch.cat([w_32k, nxt_32k[:, :need_tail]], dim=1)

    return {"waveform": pad_audio_800(w_32k.unsqueeze(0)).contiguous(), "sample_rate": AUDIO_VAE_SR}, delivery_ref_audio
