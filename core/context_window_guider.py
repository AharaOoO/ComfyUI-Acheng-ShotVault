"""Joint H3 ContextWindows (in-step X0 anchoring, _late_handover, DP speech), 1st+2nd asymmetric guides & keyframe_until."""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from ..compat import ensure_audio_feather_patch, resample_audio
from .grid_math import (
    AUDIO_LATENT_FPS,
    AUDIO_VAE_SR,
    FPS_DEFAULT,
    FRAME_PER_TOKEN,
    FRAME_RESCALE,
    GROUP_FRAMES,
    GROUP_TOKENS,
    align_frame_count,
    audio_latent_t,
    pad_audio_800,
    resolve_audio_guide_end,
    time_shift_sigma,
    video_latent_t,
)

LOGGER = logging.getLogger("ShotVault.WindowGuider")
WINDOWS_KEY = "shotvault_h3_windows"
LATE_HANDOVER_TOKENS = 2
MIN_LAST_WINDOW_FRAMES = 56
SPEECH_PAD_FRAMES = 4
SPEECH_MIN_WINDOW_GROUPS = 7
KEYFRAME_WRAPPER_KEY = "shotvault_keyframe_until"
WINDOW_ANCHOR_START_SIGMA = 0.90


@dataclass(frozen=True)
class Window:
    frame_start: int
    frames: int
    video_start: int
    video_len: int
    audio_start: int
    audio_len: int

    @property
    def video_end(self) -> int:
        return self.video_start + self.video_len

    @property
    def audio_end(self) -> int:
        return self.audio_start + self.audio_len


@dataclass
class WindowSet:
    windows: list[Window]
    conditionings: list[Any]
    total_frames: int

    @property
    def video_total(self) -> int:
        return video_latent_t(self.total_frames)

    @property
    def audio_total(self) -> int:
        return audio_latent_t(self.total_frames)


def groups_of(frames: int) -> int:
    frames = int(frames)
    if frames < 5 or (frames - 5) % GROUP_FRAMES:
        raise ValueError(f"ShotVault ContextWindows: {frames} is not 17k+5.")
    return (frames - 5) // GROUP_FRAMES


def frames_of(groups: int) -> int:
    return 5 + GROUP_FRAMES * int(groups)


def overlap_groups_of(overlap_frames: int) -> int:
    return max(1, math.ceil((int(overlap_frames) - 5) / GROUP_FRAMES))


def _audio_start(frame_start: int, frames: int, total_frames: int) -> int:
    full_a = audio_latent_t(total_frames)
    len_a = audio_latent_t(frames)
    if int(frame_start) + int(frames) >= int(total_frames):
        return max(0, full_a - len_a)
    return max(0, min(round(frame_start * FRAME_RESCALE), full_a - len_a))


def _make_window(frame_start: int, frames: int, total_frames: int) -> Window:
    return Window(
        frame_start=frame_start,
        frames=frames,
        video_start=GROUP_TOKENS * (frame_start // GROUP_FRAMES),
        video_len=video_latent_t(frames),
        audio_start=_audio_start(frame_start, frames, total_frames),
        audio_len=audio_latent_t(frames),
    )


def plan_windows(total_frames: int, window_frames: int = 192, overlap_frames: int = 39) -> list[Window]:
    """Plan 17-frame group-aligned overlapping context windows covering `total_frames` (`17k+5`)."""
    total_frames = align_frame_count(total_frames)
    total_groups = groups_of(total_frames)
    window_groups = max(2, (max(5, int(window_frames)) - 5) // GROUP_FRAMES)
    ov_groups = overlap_groups_of(overlap_frames)
    if ov_groups >= window_groups:
        raise ValueError(f"ShotVault ContextWindows: overlap {overlap_frames} must be smaller than window {window_frames}.")
    if total_groups <= window_groups:
        return [Window(0, total_frames, 0, video_latent_t(total_frames), 0, audio_latent_t(total_frames))]
    stride = window_groups - ov_groups
    shortest_last = max(groups_of(MIN_LAST_WINDOW_FRAMES), ov_groups + 1)
    spans: list[tuple[int, int]] = []
    start = 0
    while start + window_groups < total_groups:
        spans.append((start, window_groups))
        start += stride
    if total_groups - start < shortest_last:
        start = total_groups - shortest_last
        if spans:
            start = max(start, spans[-1][0] + 1)
    spans.append((start, total_groups - start))
    return [_make_window(GROUP_FRAMES * g, frames_of(cnt), total_frames) for g, cnt in spans]


def speech_in_overlaps(windows: list[Window], speech: list[tuple[int, int]]) -> int:
    frames = set()
    for before, after in zip(windows, windows[1:]):
        s, e = after.frame_start, before.frame_start + before.frames
        for a, b in speech:
            frames.update(range(max(s, int(a)), min(e, int(b))))
    return len(frames)


def plan_speech_windows(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
    speech: list[tuple[int, int]],
) -> tuple[list[Window], int]:
    """DP window planner placing overlaps into silent gaps between spoken dialogue lines."""
    total_frames = align_frame_count(total_frames)
    plain = plan_windows(total_frames, window_frames, overlap_frames)
    if len(plain) < 2 or not speech:
        return plain, 0
    total_groups = groups_of(total_frames)
    longest = groups_of(plain[0].frames)
    ov_groups = overlap_groups_of(overlap_frames)
    shortest = min(longest, max(SPEECH_MIN_WINDOW_GROUPS, ov_groups + 1))

    talking = [False] * total_frames
    for a, b in speech:
        for i in range(max(0, int(a) - SPEECH_PAD_FRAMES), min(total_frames, int(b) + SPEECH_PAD_FRAMES)):
            talking[i] = True

    costs: dict[tuple[int, int], tuple[float, int]] = {}

    def cost(end_g: int, next_start: int) -> tuple[float, int]:
        key = (end_g, next_start)
        if key not in costs:
            a, b = GROUP_FRAMES * next_start, min(total_frames, GROUP_FRAMES * end_g + 5)
            length = max(0, b - a)
            weighted, plain_f = 0.0, 0
            for idx in range(length):
                if talking[a + idx]:
                    share = (idx + 1) / (length + 1)
                    weighted += 2.0 * min(share, 1.0 - share)
                    plain_f += 1
            costs[key] = (weighted, plain_f)
        return costs[key]

    best: dict[tuple[int, int], tuple[float, int, int, int, list[tuple[int, int]]]] = {
        (0, e): (0.0, 1, -e, 0, [(0, e)]) for e in range(shortest, min(longest, total_groups) + 1)
    }
    for s in range(total_groups):
        for e in range(s + shortest, min(s + longest, total_groups - 1) + 1):
            here = best.get((s, e))
            if here is None:
                continue
            for r in range(s + 1, e - ov_groups + 1):
                w_cost, f_cost = cost(e, r)
                for e2 in range(max(e + 1, r + shortest), min(r + longest, total_groups) + 1):
                    cand = (round(here[0] + w_cost, 6), here[1] + 1, here[2] - (e2 - r), here[3] + f_cost, here[4] + [(r, e2)])
                    if (r, e2) not in best or cand[:3] < best[(r, e2)][:3]:
                        best[(r, e2)] = cand

    finals = [v for (s, e), v in best.items() if e == total_groups]
    if not finals:
        return plain, speech_in_overlaps(plain, speech)
    weighted, count, _, _, path = min(finals, key=lambda x: x[:3])
    even = [(w.frame_start // GROUP_FRAMES, groups_of(w.frames) + w.frame_start // GROUP_FRAMES) for w in plain]
    even_w = round(sum(cost(l[1], r[0])[0] for l, r in zip(even, even[1:])), 6)
    chosen = plain if (even_w <= weighted and len(even) <= count) else [
        _make_window(GROUP_FRAMES * s, frames_of(e - s), total_frames) for s, e in path
    ]
    return chosen, speech_in_overlaps(chosen, speech)


def _ramp(length: int, overlap: int, rising: bool) -> torch.Tensor:
    weights = torch.ones(length, dtype=torch.float32)
    overlap = max(0, min(int(overlap), length))
    if overlap:
        ramp = torch.arange(1, overlap + 1, dtype=torch.float32) / (overlap + 1)
        if rising:
            weights[:overlap] = ramp
        else:
            weights[length - overlap:] = ramp.flip(0)
    return weights


def _late_handover(length: int, overlap: int, rising: bool, fade: int = LATE_HANDOVER_TOKENS) -> torch.Tensor:
    """Keep earlier window's tokens across overlap except linear crossfade on the last `fade` tokens."""
    weights = torch.ones(length, dtype=torch.float32)
    overlap = max(0, min(int(overlap), length))
    if not overlap:
        return weights
    fade = max(1, min(int(fade), overlap))
    later = torch.zeros(overlap, dtype=torch.float32)
    later[overlap - fade:] = torch.arange(1, fade + 1, dtype=torch.float32) / (fade + 1)
    if rising:
        weights[:overlap] = later
    else:
        weights[length - overlap:] = 1.0 - later
    return weights


def window_weights(windows: list[Window], stream: str, late: bool = False) -> list[torch.Tensor]:
    def bounds(w: Window):
        return (w.video_start, w.video_end) if stream == "video" else (w.audio_start, w.audio_end)

    edge = _late_handover if late else _ramp
    out = []
    for i, w in enumerate(windows):
        s, e = bounds(w)
        weights = torch.ones(e - s, dtype=torch.float32)
        if i > 0:
            weights = torch.minimum(weights, edge(e - s, bounds(windows[i - 1])[1] - s, True))
        if i + 1 < len(windows):
            weights = torch.minimum(weights, edge(e - s, e - bounds(windows[i + 1])[0], False))
        out.append(weights)
    return out


def blend(parts: list[torch.Tensor], windows: list[Window], weights: list[torch.Tensor], total: int, stream: str, dim: int) -> torch.Tensor:
    first = parts[0]
    shape = list(first.shape)
    shape[dim] = int(total)
    out = torch.zeros(shape, dtype=torch.float32, device=first.device)
    counter = torch.zeros(int(total), dtype=torch.float32, device=first.device)
    view = [1] * len(shape)
    for part, w, wt in zip(parts, windows, weights):
        start = w.video_start if stream == "video" else w.audio_start
        wt = wt.to(first.device)
        view[dim] = int(wt.numel())
        out.narrow(dim, start, part.shape[dim]).add_(part.float() * wt.view(view))
        counter[start:start + part.shape[dim]] += wt
    if bool((counter <= 0).any()):
        raise ValueError("ShotVault ContextWindows: uncovered timeline tokens during blend.")
    view[dim] = int(total)
    return (out / counter.view(view)).to(first.dtype)


def decode_video_chunked(decode_fn: Callable[[torch.Tensor], torch.Tensor], video: torch.Tensor, groups_per_chunk: int = 21) -> torch.Tensor:
    """Decode long video latents (`> 21` groups) in exact 5-token chunk alignment with zero boundary error."""
    total = int(video.shape[2])
    groups = (total - 2) // GROUP_TOKENS
    if (total - 2) % GROUP_TOKENS or groups < 1 or groups <= groups_per_chunk:
        return decode_fn(video)
    chunks = []
    step = max(1, int(groups_per_chunk))
    for g in range(0, groups, step):
        cnt = min(step, groups - g)
        piece = video[:, :, GROUP_TOKENS * g:GROUP_TOKENS * (g + cnt) + 2]
        frames = decode_fn(piece)
        last = g + cnt >= groups
        chunks.append(frames if last else frames[:GROUP_FRAMES * cnt])
    return torch.cat(chunks, dim=0)


def add_window_anchor(conds: dict, before: Window, window: Window, video_shape) -> Optional[torch.Tensor]:
    """Attach zero overlap anchor at window frame 0 if no video keyframe already pins frame 0."""
    overlap = before.video_end - window.video_start
    if overlap <= 0:
        return None
    shape = (1, int(video_shape[1]), int(overlap), int(video_shape[3]), int(video_shape[4]))
    placeholder = torch.zeros(shape, dtype=torch.float32)
    added = False
    for entries in conds.values():
        for entry in entries:
            kfs = list(entry.get("minimax_keyframes") or ())
            has_video_0 = any(kf.get("latent") is not None and int(kf.get("resolved_frame_index", -1)) == 0 for kf in kfs)
            if has_video_0:
                continue
            kfs.insert(0, {"resolved_frame_index": 0, "latent": placeholder, "is_window_anchor": True})
            entry["minimax_keyframes"] = kfs
            entry["minimax_frame_count"] = int(window.frames)
            added = True
    return placeholder if added else None


def apply_window_rope_offset(conds: dict, window: Window) -> int:
    """Shift target-timeline RoPE `position_ids[:, 0]` on an immutable per-window layout clone.

    Preserves `text`, `ref_img`, and `ref_audio` coordinates at `[0..cursor)` while shifting
    `video`/`cond` by `FRAME_RESCALE * window.frame_start` and `audio`/`cond_audio` by
    `float(window.audio_start)` so later windows keep exact global timeline positions rather
    than resetting RoPE t=0. Never mutates the original layout in place.
    """
    if int(window.frame_start) <= 0 and int(window.audio_start) <= 0:
        return 0
    import copy

    delta_v = float(FRAME_RESCALE * int(window.frame_start))
    delta_a = float(int(window.audio_start))
    shifted = 0
    for entries in conds.values():
        for entry in entries:
            mc = entry.get("model_conds") or {}
            mp = mc.get("minimax_payload")
            payload = getattr(mp, "cond", None)
            if not isinstance(payload, dict):
                continue
            layout = payload.get("layout")
            pos_ids = getattr(layout, "position_ids", None)
            segments = getattr(layout, "segments", None)
            if not isinstance(pos_ids, torch.Tensor) or pos_ids.ndim != 2 or pos_ids.shape[1] < 1 or not segments:
                continue
            new_pos = pos_ids.clone()
            for a, b, kind in segments:
                if kind in ("video", "cond"):
                    new_pos[a:b, 0] = new_pos[a:b, 0] + delta_v
                elif kind in ("audio", "cond_audio"):
                    new_pos[a:b, 0] = new_pos[a:b, 0] + delta_a
            new_layout = copy.copy(layout)
            new_layout.position_ids = new_pos
            new_payload = dict(payload, layout=new_layout)
            if hasattr(mp, "_copy_with"):
                mc["minimax_payload"] = mp._copy_with(new_payload)
            else:
                mp.cond = new_payload
            shifted += 1
    return shifted


def _extract_sigma_scalar(timestep: Any) -> float:
    """Extract normalized `[0.0, 1.0]` flow sigma from `predict_noise` timestep argument."""
    try:
        raw = float(timestep.flatten()[0].item()) if isinstance(timestep, torch.Tensor) else float(timestep)
    except Exception:
        return 0.0
    return raw / 1000.0 if raw > 1.5 else raw


def anchor_latents(conds: dict, placeholder: torch.Tensor) -> list[torch.Tensor]:
    found = []
    for entries in conds.values():
        for entry in entries:
            mc = entry.get("model_conds") or {}
            payload = getattr(mc.get("minimax_payload"), "cond", None)
            if not isinstance(payload, dict):
                continue
            keyframes = payload.get("keyframes") or []
            latents = payload.get("cond_video_latents") or []
            for pos, kf in enumerate(k for k in keyframes if k.get("latent") is not None):
                if kf.get("latent") is placeholder and pos < len(latents):
                    if all(latents[pos] is not s for s in found):
                        found.append(latents[pos])
    if not found and placeholder is not None:
        found.append(placeholder)
    return found


def apply_keyframe_until(model, until_sigma: float = 0.85):
    """Release ONLY `shotvault_continuation` video keyframes below `until_sigma`; retain audio guides & window anchors."""
    until_val = float(until_sigma)
    if until_val <= 0.0:
        return model
    import comfy.patcher_extension  # type: ignore

    existing = (getattr(model, "wrappers", {}) or {}).get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
    if KEYFRAME_WRAPPER_KEY in existing:
        return model

    def release(executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        payload = minimax_payload or {}
        if payload.get("keyframes") and float(timestep.flatten()[0]) / 1000.0 < until_val:
            kept = [kf for kf in payload["keyframes"] if not kf.get("shotvault_continuation")]
            refs = payload.get("refs") or []
            cond_v = [kf["latent"] for kf in kept if kf.get("latent") is not None] + [r["latent"] for r in refs if "latent" in r]
            cond_a = [kf["audio_latent"] for kf in kept if kf.get("audio_latent") is not None] + [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
            minimax_payload = dict(
                payload,
                keyframes=kept or None,
                layout=None,
                cond_video_latents=cond_v,
                cond_audio_latents=cond_a,
            )
        return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEYFRAME_WRAPPER_KEY, release)
    return patched


def slice_keyframes_for_window(keyframes: list[dict], window: Window) -> list[dict]:
    """Slice global timeline keyframes (`video` on exact FRAME_PER_TOKEN 5-token chunks preserving negative indices)."""
    out: list[dict] = []
    w0, w1 = window.frame_start, window.frame_start + window.frames
    a0, a1 = window.audio_start, window.audio_end
    for kf in keyframes or ():
        f = float(kf.get("resolved_frame_index", 0))
        v_lat = kf.get("latent")
        a_lat = kf.get("audio_latent")
        if v_lat is not None:
            vt = int(v_lat.shape[2])
            guide_starts = [0]
            for k in range(vt):
                guide_starts.append(guide_starts[-1] + FRAME_PER_TOKEN[k % GROUP_TOKENS])
            chunks = [
                g for g in range(0, vt, GROUP_TOKENS)
                if f + guide_starts[g] < w1 and f + guide_starts[min(g + GROUP_TOKENS, vt)] > w0
            ]
            if chunks:
                first_c = chunks[0]
                last_c = min(vt, chunks[-1] + GROUP_TOKENS)
                item = dict(kf)
                item["resolved_frame_index"] = int(round(f)) + guide_starts[first_c] - w0
                item["latent"] = v_lat[:, :, first_c:last_c].clone()
                item.pop("audio_latent", None)
                out.append(item)
        if a_lat is not None:
            t_a = FRAME_RESCALE * f
            j0 = max(0, math.ceil(a0 - t_a))
            j1 = min(int(a_lat.shape[-1]), math.ceil(a1 - t_a))
            if j1 > j0:
                item = {k: v for k, v in kf.items() if k != "latent"}
                item["resolved_frame_index"] = (t_a + j0 - a0) / FRAME_RESCALE
                item["audio_latent"] = a_lat[..., j0:j1].clone()
                out.append(item)
    return out


def _attach_partial_audio_rebase_wrapper(model_options: dict, inner_model: Any) -> None:
    """Rebase unscaled audio stream (`audio_scale==1.0`) onto shift=3.0 clock during partial restarts."""
    import comfy.patcher_extension  # type: ignore

    def rebase_wrapper(executor, guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar):
        scale_fn = getattr(inner_model, "audio_scale", None)
        audio_scale = float(scale_fn()) if callable(scale_fn) else 1.0
        shapes = getattr(inner_model, "latent_shapes", None)
        sigma_v = float(sigmas[0]) if sigmas is not None and len(sigmas) > 0 else 1.0
        if (
            abs(audio_scale - 1.0) <= 1e-4
            and shapes
            and len(shapes) >= 2
            and 0.001 < sigma_v < 0.999
            and noise is not None
            and latent_image is not None
        ):
            video_values = math.prod(shapes[0][1:])
            if noise.shape[-1] > video_values and torch.count_nonzero(noise[..., video_values:]) > 0:
                sigma_a = float(time_shift_sigma(sigma_v, 12.0, 3.0))
                noise = noise.clone()
                latent_image = latent_image.clone()
                noise[..., video_values:] *= (sigma_a / sigma_v)
                latent_image[..., video_values:] *= ((1.0 - sigma_a) / max(1e-6, 1.0 - sigma_v))
        return executor(guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        "shotvault_partial_audio_rebase",
        rebase_wrapper,
        model_options,
        is_model_options=True,
    )


def make_window_guider(
    model,
    conditioning,
    windows: Optional[list[Window]],
    total_frames: int,
    cfg: float = 1.0,
    negative=None,
    pass_stage: str = "single_or_1st",
    rebase_2nd_pass_audio: bool = True,
):
    """Build `H3ContextWindowGuider` for multi-window clips, or safe `CFGGuider` (`Guider_Basic` when negative=None)."""
    import comfy.model_patcher  # type: ignore
    import comfy.patcher_extension  # type: ignore
    import comfy.sampler_helpers  # type: ignore
    import comfy.samplers  # type: ignore
    import comfy.utils  # type: ignore

    use_neg = negative is not None and float(cfg) > 1.0
    if not windows or len(windows) < 2:
        g = comfy.samplers.CFGGuider(model)
        if use_neg:
            g.set_conds(conditioning, negative)
            g.set_cfg(float(cfg))
        else:
            g.inner_set_conds({"positive": conditioning})
            g.set_cfg(1.0)
        if pass_stage == "2nd_refine" and rebase_2nd_pass_audio:
            g.model_options = comfy.model_patcher.create_model_options_clone(g.model_options)
            _attach_partial_audio_rebase_wrapper(g.model_options, getattr(model, "model", None))
        return g

    base_cond = conditioning
    window_conds = []
    for w in windows:
        c_list = []
        for emb, meta in base_cond:
            m = dict(meta or {})
            if "minimax_keyframes" in m:
                m["minimax_keyframes"] = slice_keyframes_for_window(m["minimax_keyframes"], w)
            m["minimax_frame_count"] = int(w.frames)
            c_list.append([emb, m])
        window_conds.append(c_list)
    window_set = WindowSet(list(windows), window_conds, int(total_frames))

    class H3ContextWindowGuider(comfy.samplers.CFGGuider):
        def __init__(self, model_patcher):
            super().__init__(model_patcher)
            self.model_options = comfy.model_patcher.create_model_options_clone(model_patcher.model_options)
            self.window_set = window_set
            self.pass_stage = pass_stage
            self.rebase_2nd_pass_audio = bool(rebase_2nd_pass_audio)
            self.window_raw = []
            for cond in window_set.conditionings:
                entry = {"positive": cond}
                if use_neg:
                    entry["negative"] = negative
                self.window_raw.append(entry)
            self.set_cfg(float(cfg) if use_neg else 1.0)
            self.inner_set_conds(self.window_raw[0])
            ratio = max(w.video_len for w in window_set.windows) / float(max(1, window_set.video_total))

            def window_memory(executor, mdl, noise_shape, conds, *args, **kwargs):
                shape = list(noise_shape)
                shape[-1] = max(1, int(math.ceil(int(shape[-1]) * ratio)))
                return executor(mdl, tuple(shape), conds, *args, **kwargs)

            comfy.patcher_extension.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                "shotvault_h3_context_windows",
                window_memory,
                self.model_options,
                is_model_options=True,
            )

        def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
            if latent_shapes is None or len(latent_shapes) < 2:
                raise ValueError("ShotVault ContextWindows requires MiniMax H3 AV latent.")
            self.full_shapes = latent_shapes
            video_shape, _ = latent_shapes[0], latent_shapes[1]
            self.inner_model.latent_shapes = latent_shapes
            if latent_image is not None and torch.count_nonzero(latent_image) > 0:
                latent_image = self.inner_model.process_latent_in(latent_image)

            def window_pack(packed, w: Window):
                if packed is None:
                    return None, None
                v, a = comfy.utils.unpack_latents(packed, latent_shapes)
                return comfy.utils.pack_latents([v[:, :, w.video_start:w.video_end], a[..., w.audio_start:w.audio_end]])

            self.runs = []
            self.runs_free = []
            self.anchors = []
            for idx, (w, raw) in enumerate(zip(self.window_set.windows, self.window_raw)):
                w_noise, w_shapes = window_pack(noise, w)
                w_lat, _ = window_pack(latent_image, w)
                w_msk, _ = window_pack(denoise_mask, w)
                conds_free_raw = {k: [dict(item) for item in comfy.sampler_helpers.convert_cond(val)] for k, val in raw.items()}
                conds_free = comfy.samplers.process_conds(self.inner_model, w_noise, conds_free_raw, device, w_lat, w_msk, seed, latent_shapes=w_shapes)
                apply_window_rope_offset(conds_free, w)
                if idx > 0:
                    conds_anc_raw = {k: [dict(item) for item in comfy.sampler_helpers.convert_cond(val)] for k, val in raw.items()}
                    ph = add_window_anchor(conds_anc_raw, self.window_set.windows[idx - 1], w, video_shape)
                    conds_anc = comfy.samplers.process_conds(self.inner_model, w_noise, conds_anc_raw, device, w_lat, w_msk, seed, latent_shapes=w_shapes)
                    apply_window_rope_offset(conds_anc, w)
                    self.runs.append((w, conds_anc, w_shapes))
                    self.runs_free.append((w, conds_free, w_shapes))
                    self.anchors.append(anchor_latents(conds_anc, ph) if ph is not None else [])
                else:
                    self.runs.append((w, conds_free, w_shapes))
                    self.runs_free.append((w, conds_free, w_shapes))
                    self.anchors.append([])

            self.inner_model.latent_shapes = latent_shapes
            anchored = len(self.anchors) > 1 and all(self.anchors[1:])
            self.weights_v = window_weights(self.window_set.windows, "video", late=anchored)
            self.weights_v_free = window_weights(self.window_set.windows, "video", late=False)
            self.weights_a = window_weights(self.window_set.windows, "audio")
            self.anchor_start_sigma = float(WINDOW_ANCHOR_START_SIGMA)
            self.conds = self.runs[0][1]

            extra_opts = comfy.model_patcher.create_model_options_clone(self.model_options)
            extra_opts.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
            if self.pass_stage == "2nd_refine" and self.rebase_2nd_pass_audio:
                _attach_partial_audio_rebase_wrapper(extra_opts, self.inner_model)
            extra_args = {"model_options": extra_opts, "seed": seed}
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                sampler.sample,
                sampler,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True),
            )
            with contextlib.ExitStack() as stack:
                if hasattr(self.model_patcher, "writable_models"):
                    stack.enter_context(self.model_patcher.writable_models(self.conds, noise.shape, self.model_options))
                samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
            return self.inner_model.process_latent_out(samples.to(torch.float32))

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            video, audio = comfy.utils.unpack_latents(x, self.full_shapes)
            sigma_v = _extract_sigma_scalar(timestep)
            use_anchor = sigma_v <= self.anchor_start_sigma
            active_runs = self.runs if use_anchor else self.runs_free
            active_weights_v = self.weights_v if use_anchor else self.weights_v_free
            parts_v, parts_a = [], []
            for idx, (w, conds, shapes) in enumerate(active_runs):
                if use_anchor and idx > 0 and self.anchors[idx]:
                    before = active_runs[idx - 1][0]
                    lead = parts_v[idx - 1][:, :, w.video_start - before.video_start:before.video_len]
                    for anchor in self.anchors[idx]:
                        anchor.copy_(lead.to(device=anchor.device, dtype=anchor.dtype))
                self.conds = conds
                packed, _ = comfy.utils.pack_latents([video[:, :, w.video_start:w.video_end], audio[..., w.audio_start:w.audio_end]])
                out = comfy.samplers.sampling_function(self.inner_model, packed, timestep, conds.get("negative"), conds.get("positive"), self.cfg, model_options=model_options, seed=seed)
                out_v, out_a = comfy.utils.unpack_latents(out, shapes)
                parts_v.append(out_v)
                parts_a.append(out_a)
            video = blend(parts_v, self.window_set.windows, active_weights_v, self.window_set.video_total, "video", 2)
            audio = blend(parts_a, self.window_set.windows, self.weights_a, self.window_set.audio_total, "audio", audio.ndim - 1)
            return comfy.utils.pack_latents([video, audio])[0]

    return H3ContextWindowGuider(model)


def _resize_guide_images(clip: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize `[B, H, W, 3]` to `(height, width)` via Lanczos center-crop matching `MiniMaxH3AddGuide._resize`."""
    if int(clip.shape[1]) == int(height) and int(clip.shape[2]) == int(width):
        return clip
    try:
        import comfy.utils  # type: ignore

        return comfy.utils.common_upscale(clip[..., :3].movedim(-1, 1), int(width), int(height), "lanczos", "center").movedim(1, -1)
    except Exception:
        return F.interpolate(clip[..., :3].movedim(-1, 1), size=(int(height), int(width)), mode="bilinear", align_corners=False).movedim(1, -1)


def _encode_audio_800(audio_vae, audio: dict) -> torch.Tensor:
    wav = audio["waveform"]
    sr = int(audio.get("sample_rate", AUDIO_VAE_SR))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", AUDIO_VAE_SR))
    if sr != vae_sr:
        wav = resample_audio(wav, sr, vae_sr)
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    wav = pad_audio_800(wav[:1].float())
    return audio_vae.encode(wav.movedim(1, -1))


def _encode_audio_1114_guide(audio_vae, audio: dict) -> torch.Tensor:
    """Encode 1114 continuation audio guide directly without appending a 0.0 zero-silence cliff at the tail."""
    wav = audio["waveform"]
    sr = int(audio.get("sample_rate", AUDIO_VAE_SR))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", AUDIO_VAE_SR))
    if sr != vae_sr:
        wav = resample_audio(wav, sr, vae_sr)
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    w = wav[:1].float()
    try:
        return audio_vae.encode(w.movedim(1, -1))
    except Exception:
        rem = int(w.shape[-1]) % 800
        if rem != 0 and int(w.shape[-1]) > 0:
            w = F.pad(w, (0, 800 - rem), mode="replicate")
        else:
            w = pad_audio_800(w)
        return audio_vae.encode(w.movedim(1, -1))


_REVERSE_TAPER_PALETTE = (
    (185.0 / 255.0, 115.0 / 255.0, 215.0 / 255.0),
    (115.0 / 255.0, 195.0 / 255.0, 140.0 / 255.0),
    (150.0 / 255.0, 148.0 / 255.0, 162.0 / 255.0),
    (205.0 / 255.0, 150.0 / 255.0, 192.0 / 255.0),
    (138.0 / 255.0, 182.0 / 255.0, 148.0 / 255.0),
    (160.0 / 255.0, 120.0 / 255.0, 175.0 / 255.0),
)


def _apply_reverse_taper_noise(
    frames_hwc: torch.Tensor,
    alpha: float = 0.25,
    ramp_frames: int = 4,
    seed: int = 730009,
    block_size: int = 16,
) -> torch.Tensor:
    """Mirror-direction mild chroma-block taper for 22f tail anchors.

    Position 0 touches the seam with the generated shot and is strictly kept at `alpha == 0.0`
    (100% clean); positions `1..ramp_frames-1` ramp linearly up to `alpha`; positions `>= ramp_frames`
    sit at mild `alpha` (default 0.25). Never used for 1f or 5f tail anchors.
    """
    f_cnt, h, w = int(frames_hwc.shape[0]), int(frames_hwc.shape[1]), int(frames_hwc.shape[2])
    if f_cnt < 5 or alpha <= 1e-4:
        return frames_hwc
    out = frames_hwc.clone()
    dev = out.device
    dt = out.dtype
    palette = torch.tensor(_REVERSE_TAPER_PALETTE, device=dev, dtype=dt)
    gh = max(1, (h + block_size - 1) // block_size)
    gw = max(1, (w + block_size - 1) // block_size)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    ramp = max(1, min(int(ramp_frames), f_cnt - 1))
    for pos in range(f_cnt):
        if pos == 0:
            a = 0.0
        elif pos < ramp:
            a = float(alpha) * (float(pos) / float(ramp))
        else:
            a = float(alpha)
        if a <= 1e-6:
            continue
        idx_grid = torch.randint(0, len(_REVERSE_TAPER_PALETTE), (gh, gw), generator=gen).to(dev)
        blocks = palette[idx_grid].permute(2, 0, 1).unsqueeze(0)  # [1, 3, gh, gw]
        up = F.interpolate(blocks, size=(h, w), mode="nearest").squeeze(0).permute(1, 2, 0)
        out[pos] = torch.lerp(out[pos], up, a).clamp(0.0, 1.0)
    return out


def _apply_anti_degradation_guide(
    tapered_clip: torch.Tensor,
    clean_tail: Optional[torch.Tensor],
    scene_anchor: Optional[torch.Tensor],
    anti_deg_cfg: Optional[dict[str, Any]],
    continuation_depth: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Attenuate fixed-camera chroma-noise snowball & optionally refresh verified static background from Shot 1 anchor.

    - `standard`: bit-for-bit passthrough of `tapered_clip` (zero overhead, zero change for action / multi-cut workflows).
    - `digital_human_low_noise`: linearly pulls `tapered_clip` toward `clean_tail` by `noise_keep_ratio` (default 0.40 -> ~0.22 eff alpha).
    - `digital_human_static_bg`: low-noise pull + 3-gate static-background anchor refresh (foreground motion protection,
      pixel anchor consistency, and global fixed-camera correlation gate that auto-bypasses on camera motion to prevent ghosting).
    """
    cfg = anti_deg_cfg if isinstance(anti_deg_cfg, dict) else {}
    mode = str(cfg.get("mode") or "standard").strip()
    if mode not in ("digital_human_low_noise", "digital_human_static_bg"):
        return tapered_clip, {"mode": "standard", "applied": "passthrough"}
    if not isinstance(clean_tail, torch.Tensor) or int(clean_tail.shape[0]) < 1:
        return tapered_clip, {"mode": mode, "applied": "fallback_no_clean_tail"}

    f_cnt, h, w = int(tapered_clip.shape[0]), int(tapered_clip.shape[1]), int(tapered_clip.shape[2])
    c_src = clean_tail[-min(f_cnt, int(clean_tail.shape[0])):, ..., :3].to(device=tapered_clip.device, dtype=tapered_clip.dtype)
    if int(c_src.shape[0]) < f_cnt:
        pad_n = f_cnt - int(c_src.shape[0])
        c_src = torch.cat([c_src[:1].repeat(pad_n, 1, 1, 1), c_src], dim=0)
    clean_clip = _resize_guide_images(c_src, w, h)

    def_lam = 0.36 if mode == "digital_human_static_bg" else 0.40
    try:
        lam = max(0.05, min(1.0, float(cfg.get("noise_keep_ratio", def_lam))))
    except (TypeError, ValueError):
        lam = def_lam
    eff = torch.lerp(clean_clip, tapered_clip, lam).clamp(0.0, 1.0)
    info: dict[str, Any] = {
        "mode": mode,
        "noise_keep_ratio": round(lam, 4),
        "continuation_depth": int(continuation_depth),
        "applied": "low_noise_only",
    }

    if mode != "digital_human_static_bg" or int(continuation_depth) < 2:
        return eff, info
    if not isinstance(scene_anchor, torch.Tensor) or int(scene_anchor.shape[0]) < 1 or f_cnt < 2:
        info["static_bg_skipped"] = "no_scene_anchor"
        return eff, info

    try:
        beta = max(0.0, min(0.35, float(cfg.get("static_bg_pull", 0.20))))
    except (TypeError, ValueError):
        beta = 0.20
    if beta <= 1e-4:
        return eff, info

    anc_src = scene_anchor[-min(4, int(scene_anchor.shape[0])):, ..., :3].to(device=eff.device, dtype=eff.dtype)
    anc_img = _resize_guide_images(anc_src, w, h).mean(dim=0)  # [H, W, 3]
    clean_mean = clean_clip.mean(dim=0)  # [H, W, 3]

    # Gate 1: Foreground motion mask (dilated & feathered to protect face, lips, eyes, hair, hands)
    mot = (clean_clip - clean_mean.unsqueeze(0)).abs().amax(dim=(0, -1))
    fg_raw = ((mot - 0.025) / (0.095 - 0.025)).clamp(0.0, 1.0)
    rad = max(5, min(25, w // 32)) | 1
    b_k = max(5, (rad // 2) | 1)
    fg_dil = F.max_pool2d(fg_raw.unsqueeze(0).unsqueeze(0), kernel_size=rad, stride=1, padding=rad // 2)
    m_fg = F.avg_pool2d(fg_dil, kernel_size=b_k, stride=1, padding=b_k // 2).squeeze(0).squeeze(0).clamp(0.0, 1.0)

    # Gate 2: Global fixed-camera correlation + local pixel consistency gate (auto-bypasses on camera movement)
    d_anc = (clean_mean - anc_img).abs().amax(dim=-1)
    cand = m_fg < 0.15
    cand_frac = float(cand.float().mean().item())
    info["static_candidate_frac"] = round(cand_frac, 4)
    if cand_frac < 0.15:
        info["static_bg_skipped"] = "low_static_area"
        LOGGER.info("ShotVault anti-degradation static_bg skipped: low_static_area (%.2f)", cand_frac)
        return eff, info

    med_diff = float(torch.median(d_anc[cand]).item())
    info["anchor_median_diff"] = round(med_diff, 4)
    if med_diff > 0.10:
        info["static_bg_skipped"] = "camera_moved"
        LOGGER.info("ShotVault anti-degradation static_bg skipped: camera_moved (med_diff=%.4f)", med_diff)
        return eff, info

    # Gate 3: Bounded, feathered static-background pull toward clean Shot 1 anchor
    g_pix = ((0.14 - d_anc) / (0.14 - 0.04)).clamp(0.0, 1.0)
    m_stat_raw = ((1.0 - m_fg) * g_pix).unsqueeze(0).unsqueeze(0)
    m_stat = F.avg_pool2d(m_stat_raw, kernel_size=b_k, stride=1, padding=b_k // 2).squeeze(0).squeeze(0).clamp(0.0, 1.0)
    pull = (beta * m_stat).unsqueeze(0).unsqueeze(-1)
    out = (eff * (1.0 - pull) + anc_img.unsqueeze(0) * pull).clamp(0.0, 1.0)
    info["applied"] = "low_noise_and_static_bg_anchor"
    info["static_bg_pull"] = round(beta, 4)
    info["effective_bg_coverage"] = round(float(m_stat.mean().item()), 4)
    return out, info


def build_window_guider_and_latent(
    model,
    positive,
    latent: dict,
    vae=None,
    audio_vae=None,
    tapered_prev_images: Optional[torch.Tensor] = None,
    prev_audio: Optional[dict] = None,
    prev_tail_latent: Optional[dict] = None,
    shot_audio_guide: Optional[dict] = None,
    shot_pipe: Optional[dict] = None,
    pass_stage: str = "single_or_1st",
    window_mode: str = "auto_over_192f",
    window_frames: int = 192,
    window_overlap: int = 39,
    continuation_frames: str = "22+5",
    keyframe_until_sigma: float = 0.85,
    lock_target_audio: bool = False,
    cfg: float = 1.0,
    negative=None,
    audio_feather_ticks: int = 7,
    rebase_2nd_pass_audio: bool = True,
):
    """Prepare asymmetric 1st+2nd H3 guides, end-aligned 800-sample audio anchors, and ContextWindows guider."""
    import comfy.nested_tensor  # type: ignore

    samples = latent["samples"]
    v_lat, a_lat = samples.unbind()
    total_video_t = int(v_lat.shape[2])
    total_frames = align_frame_count(5 + 17 * ((total_video_t - 2) // 5))
    width, height = int(v_lat.shape[4]) * 16, int(v_lat.shape[3]) * 16
    target_h_lat, target_w_lat = int(v_lat.shape[3]), int(v_lat.shape[4])

    pipe = shot_pipe if isinstance(shot_pipe, dict) else {}
    if prev_audio is None and pipe:
        prev_audio = pipe.get("prev_audio") or pipe.get("parent_audio_32k_ref")
    if prev_tail_latent is None and pipe:
        prev_tail_latent = pipe.get("prev_tail_latent")
    if shot_audio_guide is None and pipe:
        shot_audio_guide = pipe.get("shot_audio_guide")

    overlap_frames = int(pipe.get("overlap_frames") or 0)
    if pipe:
        is_cont = bool(pipe.get("is_continuation", False))
    else:
        has_real_prev = (
            prev_tail_latent is not None
            or (tapered_prev_images is not None and int(tapered_prev_images.shape[0]) >= 5)
        )
        is_cont = has_real_prev

    auto_ov = overlap_frames if overlap_frames > 0 else int(pipe.get("context_frames") or 22)
    raw_parts = [x.strip().lower() for x in str(continuation_frames).split("+")]
    p0 = raw_parts[0] if raw_parts else "22"
    p1 = raw_parts[1] if len(raw_parts) > 1 else p0
    first_f = auto_ov if p0 in ("auto", "") else int(p0)
    second_f = auto_ov if p1 == "auto" else int(p1)
    active_f = first_f if pass_stage == "single_or_1st" else second_f
    if overlap_frames <= 0 and active_f > 0:
        overlap_frames = max(first_f, active_f)

    new_kfs: list[dict] = []
    symmetric_cont = first_f == second_f
    shot_idx_key = int(pipe.get("current_shot_index", pipe.get("shot_index", -1)))
    has_tapered_px = tapered_prev_images is not None and int(tapered_prev_images.shape[0]) >= 5
    anti_deg_cfg = pipe.get("anti_degradation") if isinstance(pipe.get("anti_degradation"), dict) else {}
    ad_mode = str(anti_deg_cfg.get("mode") or "standard").strip()
    ad_noise = round(float(anti_deg_cfg.get("noise_keep_ratio", 0.36 if ad_mode == "digital_human_static_bg" else 0.40)), 3)
    ad_pull = round(float(anti_deg_cfg.get("static_bg_pull", 0.20)), 3)
    cont_depth = int(pipe.get("continuation_depth") or 1)
    if is_cont and active_f > 0:
        need_steps = video_latent_t(active_f)
        cache_key_v = (
            shot_idx_key,
            str(pipe.get("parent_take_id") or ""),
            active_f,
            target_h_lat,
            target_w_lat,
            has_tapered_px,
            ad_mode,
            ad_noise,
            ad_pull,
            cont_depth,
        )
        v_guide_lat = None
        if symmetric_cont:
            cached_v = pipe.get("_cached_head_video_latent")
            if pass_stage == "2nd_refine" and isinstance(cached_v, tuple) and len(cached_v) == 2 and cached_v[0] == cache_key_v:
                v_guide_lat = cached_v[1].clone()
            else:
                if has_tapered_px:
                    if vae is None:
                        raise ValueError("ShotVaultWindowGuider: `vae` is required to encode tapered_prev_images.")
                    clip = _resize_guide_images(tapered_prev_images[-min(active_f, int(tapered_prev_images.shape[0])):, ..., :3].float(), width, height)
                    clip, ad_info = _apply_anti_degradation_guide(
                        clip,
                        pipe.get("parent_clean_tail_ref"),
                        pipe.get("scene_anchor_clean_ref"),
                        anti_deg_cfg,
                        continuation_depth=cont_depth,
                    )
                    if isinstance(shot_pipe, dict):
                        pipe["_last_anti_degradation_info"] = ad_info
                    v_guide_lat = vae.encode(clip)
                elif isinstance(prev_tail_latent, dict) and "samples" in prev_tail_latent:
                    try:
                        p_v, _ = prev_tail_latent["samples"].unbind()
                        if p_v.shape[2] >= need_steps and tuple(p_v.shape[-2:]) == (target_h_lat, target_w_lat):
                            v_guide_lat = p_v[:, :, -need_steps:].clone()
                    except Exception:
                        v_guide_lat = None
                if v_guide_lat is None and isinstance(pipe.get("parent_clean_tail_ref"), torch.Tensor):
                    p_ref = pipe["parent_clean_tail_ref"]
                    if int(p_ref.shape[0]) >= 5:
                        if vae is None:
                            raise ValueError("ShotVaultWindowGuider: `vae` is required to encode parent_clean_tail_ref.")
                        clip = _resize_guide_images(p_ref[-min(active_f, int(p_ref.shape[0])):, ..., :3].float(), width, height)
                        v_guide_lat = vae.encode(clip)
                if v_guide_lat is not None and isinstance(shot_pipe, dict):
                    pipe["_cached_head_video_latent"] = (cache_key_v, v_guide_lat.clone())
        else:
            if isinstance(prev_tail_latent, dict) and "samples" in prev_tail_latent:
                try:
                    p_v, _ = prev_tail_latent["samples"].unbind()
                    if pass_stage == "2nd_refine" or not has_tapered_px:
                        if p_v.shape[2] >= need_steps and tuple(p_v.shape[-2:]) == (target_h_lat, target_w_lat):
                            v_guide_lat = p_v[:, :, -need_steps:].clone()
                except Exception:
                    v_guide_lat = None

            if v_guide_lat is None and has_tapered_px:
                if vae is None:
                    raise ValueError("ShotVaultWindowGuider: `vae` is required to encode tapered_prev_images.")
                clip = _resize_guide_images(tapered_prev_images[-min(active_f, int(tapered_prev_images.shape[0])):, ..., :3].float(), width, height)
                clip, ad_info = _apply_anti_degradation_guide(
                    clip,
                    pipe.get("parent_clean_tail_ref"),
                    pipe.get("scene_anchor_clean_ref"),
                    anti_deg_cfg,
                    continuation_depth=cont_depth,
                )
                if isinstance(shot_pipe, dict):
                    pipe["_last_anti_degradation_info"] = ad_info
                v_guide_lat = vae.encode(clip)

        if v_guide_lat is not None:
            new_kfs.append({
                "resolved_frame_index": max(0, overlap_frames - active_f),
                "latent": v_guide_lat,
                "shotvault_continuation": True,
            })

    out_latent = dict(latent)
    audio_locked_wav = None
    has_shot_aud = (
        isinstance(shot_audio_guide, dict)
        and isinstance(shot_audio_guide.get("waveform"), torch.Tensor)
        and shot_audio_guide["waveform"].abs().max().item() > 0
    )
    has_1114_prev_aud = (
        isinstance(prev_audio, dict)
        and isinstance(prev_audio.get("waveform"), torch.Tensor)
        and audio_vae is not None
        and prev_audio["waveform"].abs().max().item() > 0
    )
    ak_key = (
        shot_idx_key,
        str(pipe.get("parent_take_id") or ""),
        overlap_frames,
        int(a_lat.shape[-1]),
        bool(has_shot_aud),
        bool(lock_target_audio),
        bool(has_tapered_px),
        int(audio_feather_ticks if audio_feather_ticks is not None else pipe.get("audio_feather_ticks", 7)),
    )
    cached_a = pipe.get("_cached_head_audio_kfs") if symmetric_cont else None
    audio_kfs_to_merge: list[dict] = []
    if symmetric_cont and pass_stage == "2nd_refine" and isinstance(cached_a, tuple) and len(cached_a) == 3 and cached_a[0] == ak_key:
        for akf in cached_a[1]:
            item_a = dict(akf)
            if isinstance(item_a.get("audio_latent"), torch.Tensor):
                item_a["audio_latent"] = item_a["audio_latent"].clone()
            audio_kfs_to_merge.append(item_a)
            if int(item_a.get("audio_feather_ticks") or 0) > 0:
                ensure_audio_feather_patch()
        if cached_a[2] is not None:
            audio_locked_wav = cached_a[2].to(a_lat)
    else:
        audio_kfs_added: list[dict] = []
        if has_shot_aud:
            if audio_vae is None:
                raise ValueError("ShotVaultWindowGuider: `audio_vae` is required when `shot_audio_guide` is connected.")
            a_enc = _encode_audio_800(audio_vae, shot_audio_guide)[..., :int(a_lat.shape[-1])]
            if a_enc.shape[-1] < int(a_lat.shape[-1]):
                a_enc = F.pad(a_enc, (0, int(a_lat.shape[-1]) - int(a_enc.shape[-1])))
            akf = {"resolved_frame_index": 0, "audio_latent": a_enc.clone(), "audio_feather_ticks": 0}
            audio_kfs_to_merge.append(akf)
            audio_kfs_added.append(akf)
            if lock_target_audio:
                audio_locked_wav = a_enc.to(a_lat)
        elif is_cont and overlap_frames > 0:
            feather_ticks = int(audio_feather_ticks if audio_feather_ticks is not None else pipe.get("audio_feather_ticks", 7))
            if pipe.get("auto_scale_feather") and audio_feather_ticks == 7 and int(pipe.get("gear_feather_ticks") or 0) > 0:
                feather_ticks = int(pipe["gear_feather_ticks"])
            ag_frames = int(pipe.get("audio_guide_frames") or max(1, overlap_frames - 2))
            prefer_1114_audio = has_1114_prev_aud and (has_tapered_px or prev_tail_latent is None)
            if prefer_1114_audio:
                max_tok = min(int(a_lat.shape[-1]), audio_latent_t(ag_frames))
                a_enc = _encode_audio_1114_guide(audio_vae, prev_audio)
                if a_enc.shape[-1] > max_tok:
                    a_enc = a_enc[..., :max_tok]
                akf = {
                    "resolved_frame_index": 0,
                    "audio_latent": a_enc.clone(),
                    "audio_feather_ticks": feather_ticks,
                }
                audio_kfs_to_merge.append(akf)
                audio_kfs_added.append(akf)
                if feather_ticks > 0:
                    ensure_audio_feather_patch()
            elif isinstance(prev_tail_latent, dict) and "samples" in prev_tail_latent:
                try:
                    _, p_a = prev_tail_latent["samples"].unbind()
                    oh = float((prev_tail_latent.get("shotvault_meta") or {}).get("audio_overhang", 0.0))
                    end_a = min(int(p_a.shape[-1]), resolve_audio_guide_end(overlap_frames, oh))
                    if end_a > 0:
                        akf = {
                            "resolved_frame_index": 0,
                            "audio_latent": p_a[..., -end_a:].clone(),
                            "audio_feather_ticks": feather_ticks,
                        }
                        audio_kfs_to_merge.append(akf)
                        audio_kfs_added.append(akf)
                        if feather_ticks > 0:
                            ensure_audio_feather_patch()
                except Exception:
                    pass
            elif has_1114_prev_aud:
                max_tok = min(int(a_lat.shape[-1]), audio_latent_t(ag_frames))
                a_enc = _encode_audio_1114_guide(audio_vae, prev_audio)
                if a_enc.shape[-1] > max_tok:
                    a_enc = a_enc[..., :max_tok]
                akf = {
                    "resolved_frame_index": 0,
                    "audio_latent": a_enc.clone(),
                    "audio_feather_ticks": feather_ticks,
                }
                audio_kfs_to_merge.append(akf)
                audio_kfs_added.append(akf)
                if feather_ticks > 0:
                    ensure_audio_feather_patch()
        if symmetric_cont and isinstance(shot_pipe, dict):
            pipe["_cached_head_audio_kfs"] = (
                ak_key,
                [dict(k, audio_latent=k["audio_latent"].clone()) for k in audio_kfs_added],
                audio_locked_wav.clone() if audio_locked_wav is not None else None,
            )

    for akf in audio_kfs_to_merge:
        if (
            len(new_kfs) == 1
            and int(new_kfs[0].get("resolved_frame_index", -1)) == int(akf.get("resolved_frame_index", -2))
            and "audio_latent" not in new_kfs[0]
        ):
            new_kfs[0]["audio_latent"] = akf["audio_latent"]
            new_kfs[0]["audio_feather_ticks"] = int(akf.get("audio_feather_ticks", 0))
        else:
            new_kfs.append(akf)

    # Bidirectional Bridge Reroll: inject clean next-shot head anchor at `tail_bridge_index` (1f / 5f / 22f)
    tb_frames = int(pipe.get("tail_bridge_frames") or 0)
    tb_index = int(pipe.get("tail_bridge_index", -1))
    next_head_img = pipe.get("next_head_clean_ref")
    if (
        tb_frames in (1, 5, 22)
        and isinstance(next_head_img, torch.Tensor)
        and int(next_head_img.shape[0]) >= tb_frames
        and tb_index >= max(overlap_frames, 1)
        and (tb_index + tb_frames) <= total_frames
    ):
        tb_noise = str(pipe.get("tail_bridge_noise") or "clean").strip()
        next_tid_key = str(pipe.get("bridged_next_take_id") or "")
        tail_ckey = (shot_idx_key, next_tid_key, tb_frames, tb_index, tb_noise, target_h_lat, target_w_lat)
        cached_tv = pipe.get("_cached_tail_video_latent")
        v_tail_lat = None
        if isinstance(cached_tv, tuple) and len(cached_tv) == 2 and cached_tv[0] == tail_ckey:
            v_tail_lat = cached_tv[1].clone()
        else:
            if vae is None:
                raise ValueError("ShotVaultWindowGuider: `vae` is required to encode bidirectional tail bridge frames.")
            t_clip = _resize_guide_images(next_head_img[:tb_frames, ..., :3].float(), width, height)
            if tb_frames == 22 and tb_noise == "reverse_taper":
                t_clip = _apply_reverse_taper_noise(t_clip, alpha=0.25, ramp_frames=4, seed=730009 + shot_idx_key)
            v_tail_lat = vae.encode(t_clip)
            if isinstance(shot_pipe, dict):
                pipe["_cached_tail_video_latent"] = (tail_ckey, v_tail_lat.clone())
        if v_tail_lat is not None:
            new_kfs.append({
                "resolved_frame_index": tb_index,
                "latent": v_tail_lat,
                "shotvault_tail_bridge": True,
            })
        # Optional 22f tail audio anchor (only when NOT in audio-driven mode where full track is already locked)
        next_head_aud = pipe.get("next_head_audio_32k")
        if (
            tb_frames == 22
            and not has_shot_aud
            and audio_vae is not None
            and isinstance(next_head_aud, dict)
            and isinstance(next_head_aud.get("waveform"), torch.Tensor)
            and next_head_aud["waveform"].abs().max().item() > 0
        ):
            aud_idx = tb_index + 2
            max_rt = math.floor(int(a_lat.shape[-1]) - FRAME_RESCALE * float(aud_idx))
            if max_rt >= 2:
                tail_akey = (shot_idx_key, next_tid_key, aud_idx, max_rt)
                cached_ta = pipe.get("_cached_tail_audio_latent")
                a_tail_enc = None
                if isinstance(cached_ta, tuple) and len(cached_ta) == 2 and cached_ta[0] == tail_akey:
                    a_tail_enc = cached_ta[1].clone()
                else:
                    a_tail_enc = _encode_audio_1114_guide(audio_vae, next_head_aud)
                    if int(a_tail_enc.shape[-1]) > max_rt:
                        a_tail_enc = a_tail_enc[..., :max_rt].clone()
                    if isinstance(shot_pipe, dict):
                        pipe["_cached_tail_audio_latent"] = (tail_akey, a_tail_enc.clone())
                if a_tail_enc is not None and int(a_tail_enc.shape[-1]) > 0:
                    new_kfs.append({
                        "resolved_frame_index": aud_idx,
                        "audio_latent": a_tail_enc,
                        "shotvault_tail_bridge": True,
                    })

    if audio_locked_wav is not None:
        out_latent["samples"] = comfy.nested_tensor.NestedTensor((v_lat, audio_locked_wav))
        out_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(v_lat), torch.zeros_like(audio_locked_wav)))
    elif pass_stage == "2nd_refine" and "noise_mask" in out_latent:
        if lock_target_audio:
            try:
                _, existing_amask = out_latent["noise_mask"].unbind()
                out_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(v_lat), existing_amask))
            except Exception:
                pass
        else:
            out_latent.pop("noise_mask", None)

    updated_pos = []
    for emb, meta in positive:
        m = dict(meta or {})
        existing = [dict(k) for k in (m.get("minimax_keyframes") or ())]
        if is_cont:
            existing = [k for k in existing if int(k.get("resolved_frame_index", 0)) >= overlap_frames]
        combined = [*new_kfs, *existing]
        if combined:
            m["minimax_keyframes"] = combined
        updated_pos.append([emb, m])

    if window_mode == "force_on":
        use_windows, eff_win_frames = True, int(window_frames)
    elif window_mode == "auto_over_260f":
        eff_win_frames = max(260, int(window_frames))
        use_windows = total_frames > eff_win_frames
    elif window_mode == "auto_over_192f":
        eff_win_frames = int(window_frames)
        use_windows = total_frames > eff_win_frames
    else:
        use_windows, eff_win_frames = False, int(window_frames)
    windows: Optional[list[Window]] = None
    if use_windows:
        raw_speech = list(pipe.get("speech_spans") or ())
        clamped_speech = [
            (max(0, min(total_frames, int(s))), max(0, min(total_frames, int(e))))
            for s, e in raw_speech
            if min(total_frames, int(e)) > max(0, int(s))
        ]
        if clamped_speech:
            windows, _ = plan_speech_windows(total_frames, eff_win_frames, int(window_overlap), clamped_speech)
        else:
            windows = plan_windows(total_frames, eff_win_frames, int(window_overlap))

    active_model = apply_keyframe_until(model, float(keyframe_until_sigma)) if model is not None else None
    guider = (
        make_window_guider(
            active_model,
            updated_pos,
            windows,
            total_frames,
            cfg=cfg,
            negative=negative,
            pass_stage=pass_stage,
            rebase_2nd_pass_audio=rebase_2nd_pass_audio,
        )
        if active_model is not None
        else updated_pos
    )
    return guider, out_latent
