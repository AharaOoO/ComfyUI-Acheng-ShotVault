"""MiniMax-H3 temporal grid math, 800-sample audio alignment, 40Hz overhang, and cumulative MV cursor."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

GROUP_FRAMES = 17
GROUP_TOKENS = 5
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS_DEFAULT = 24
AUDIO_VAE_SR = 32000
DELIVERY_AUDIO_SR = 48000
AUDIO_LATENT_FPS = 40
FRAME_RESCALE = 40.0 / 24.0
AUDIO_VAE_ALIGN = 800
VALID_CONTEXT_FRAMES = (5, 22, 39, 56)

ASPECT_PRESETS = {
    "16:9 (1344x768)": (1344, 768),
    "9:16 (768x1344)": (768, 1344),
    "2.39:1 (1536x640)": (1536, 640),
    "4:3 (1152x864)": (1152, 864),
    "3:4 (864x1152)": (864, 1152),
    "1:1 (1024x1024)": (1024, 1024),
}


def align_frame_count(n: int) -> int:
    """Return smallest valid H3 pixel frame count `17k + 5 >= max(5, n)`."""
    n = max(5, int(n))
    rem = (n - 5) % GROUP_FRAMES
    return n if rem == 0 else n + (GROUP_FRAMES - rem)


def video_latent_t(frame_count: int) -> int:
    """Convert H3 pixel frame count `17k + 5` into latent step count `5k + 2`."""
    frames = align_frame_count(frame_count)
    return 2 if frames <= 5 else ((frames - 5) // GROUP_FRAMES) * GROUP_TOKENS + 2


def audio_latent_t(frame_count: int, fps: float = FPS_DEFAULT) -> int:
    """Audio latent step count at 40 Hz matching `nodes_minimax_h3.temporal_shape`."""
    return max(1, round(float(frame_count) / float(fps) * AUDIO_LATENT_FPS))


def latent_pixel_frames(latent_t: int) -> int:
    """Compute exact decoded pixel frames for `latent_t` latent tokens."""
    return sum(FRAME_PER_TOKEN[k % GROUP_TOKENS] for k in range(max(0, int(latent_t))))


def validate_tail_slice(total_latent_t: int, tail_frames: int) -> tuple[int, int]:
    """Validate and compute `(start_step, slice_steps)` preserving `cycle_pos == 0`.

    Rejects invalid tail frame counts such as 17 (`start % 5 != 0`).
    """
    total_latent_t = int(total_latent_t)
    tail_frames = int(tail_frames)
    if tail_frames not in VALID_CONTEXT_FRAMES:
        raise ValueError(
            f"ShotVault: tail_frames={tail_frames} is not on the valid H3 continuation grid {VALID_CONTEXT_FRAMES}; "
            "slicing non-grid counts (like 17) breaks cycle_pos==0 phase alignment."
        )
    if total_latent_t < 2 or (total_latent_t - 2) % GROUP_TOKENS != 0:
        raise ValueError(f"ShotVault: source latent_t={total_latent_t} is not 5m + 2.")
    steps = ((tail_frames - 5) // GROUP_FRAMES) * GROUP_TOKENS + 2
    start = total_latent_t - steps
    if start < 0:
        raise ValueError(
            f"ShotVault: source latent_t={total_latent_t} is shorter than requested tail steps={steps}."
        )
    if start % GROUP_TOKENS != 0:
        raise AssertionError(f"ShotVault: slice start={start} violates cycle_pos==0.")
    return start, steps


def pad_audio_800(waveform: torch.Tensor) -> torch.Tensor:
    """Right-pad 32kHz audio waveform along the last axis to an 800-sample multiple."""
    length = int(waveform.shape[-1])
    rem = (-length) % AUDIO_VAE_ALIGN
    if rem == 0:
        return waveform
    return F.pad(waveform, (0, rem))


def compute_audio_overhang(audio_t: int, pixel_frames: int) -> float:
    """Compute 40Hz sub-frame overhang `T_audio - F_prev * (40/24)` clamped to `[-0.5, 0.5]`."""
    oh = float(int(audio_t)) - float(int(pixel_frames)) * FRAME_RESCALE
    return max(-0.5, min(0.5, oh))


def resolve_audio_guide_end(overlap_frames: int, overhang: float = 0.0) -> int:
    """Restore exact 40Hz audio token end index across `.h3ctx` boundaries."""
    return max(1, round(float(overlap_frames) * FRAME_RESCALE + float(overhang)))


def time_shift_sigma(sigma, shift_from: float = 12.0, shift_to: float = 3.0):
    """Rebase flow sigma from `shift_from` schedule (video=12) to `shift_to` (audio=3)."""
    s1 = float(shift_from)
    s2 = float(shift_to)
    if isinstance(sigma, torch.Tensor):
        denom = (s1 - (s1 - 1.0) * sigma).clamp(min=1e-6)
        t = (sigma / denom).clamp(0.0, 1.0)
        return (s2 * t) / (1.0 + (s2 - 1.0) * t).clamp(min=1e-6)
    s = max(0.0, min(1.0, float(sigma)))
    t = max(0.0, min(1.0, s / max(1e-6, s1 - (s1 - 1.0) * s)))
    return (s2 * t) / max(1e-6, 1.0 + (s2 - 1.0) * t)


def adapt_canvas_from_ratio(aspect_ratio: str, megapixels: float = 1.0) -> tuple[int, int]:
    """Compute `(width, height)` snapped to 32px multiples scaled from H3 reference presets."""
    mp = max(0.25, float(megapixels))
    if aspect_ratio in ASPECT_PRESETS:
        bw, bh = ASPECT_PRESETS[aspect_ratio]
    else:
        ratio = 16.0 / 9.0
        parts = str(aspect_ratio).split(" ")[0].split(":")
        if len(parts) == 2:
            try:
                rw, rh = float(parts[0]), float(parts[1])
                if rw > 0 and rh > 0:
                    ratio = rw / rh
            except ValueError:
                pass
        base_area = 1344.0 * 768.0
        bh = math.sqrt(base_area / ratio)
        bw = bh * ratio
    scale = math.sqrt(mp)
    w = max(64, round((bw * scale) / 32.0) * 32)
    h = max(64, round((bh * scale) / 32.0) * 32)
    return int(w), int(h)


_GEAR_SPEC = {
    22: {"taper_ramp_frames": 4, "audio_guide_frames": 20, "audio_feather_ticks": 7, "continuation_pair": "22+5", "video_latent_steps": 7},
    39: {"taper_ramp_frames": 6, "audio_guide_frames": 37, "audio_feather_ticks": 12, "continuation_pair": "39+5", "video_latent_steps": 12},
    56: {"taper_ramp_frames": 8, "audio_guide_frames": 54, "audio_feather_ticks": 16, "continuation_pair": "56+5", "video_latent_steps": 17},
}


def compute_overlap_gear(overlap_frames: int, base_gear: int = 22, is_continuation: bool = True) -> dict:
    """Compute synchronized downstream gear parameters for `22`, `39`, or `56` H3 overlap frames."""
    base = int(base_gear)
    if base not in (22, 39, 56):
        raise ValueError(f"ShotVault: base_gear must be 22, 39, or 56, got {base_gear}.")
    ov = int(overlap_frames)
    if not is_continuation or ov <= 0:
        safe_tail = max(22, base)
        spec_base = _GEAR_SPEC[safe_tail]
        return {
            "overlap_gear": base,
            "overlap_frames": 0,
            "trim_head_frames": 0,
            "taper_tail_frames": safe_tail,
            "taper_ramp_frames": spec_base["taper_ramp_frames"],
            "audio_guide_frames": 0,
            "audio_feather_ticks": 0,
            "continuation_pair": spec_base["continuation_pair"],
            "video_latent_steps": 0,
        }
    if ov not in (22, 39, 56):
        raise ValueError(f"ShotVault: continuation overlap_frames must be 22, 39, or 56, got {overlap_frames}.")
    spec = _GEAR_SPEC[ov]
    return {
        "overlap_gear": ov,
        "overlap_frames": ov,
        "trim_head_frames": ov,
        "taper_tail_frames": ov,
        "taper_ramp_frames": spec["taper_ramp_frames"],
        "audio_guide_frames": spec["audio_guide_frames"],
        "audio_feather_ticks": spec["audio_feather_ticks"],
        "continuation_pair": spec["continuation_pair"],
        "video_latent_steps": spec["video_latent_steps"],
    }


def resolve_duration_mode(duration_mode: str, audio_drive_active: bool = False) -> str:
    """Resolve schedule duration mode; Audio-MV mode promotes `zero_tail_trim` to beat-locked `cumulative_cursor`."""
    mode = str(duration_mode or "cumulative_cursor")
    if audio_drive_active and mode == "zero_tail_trim":
        return "cumulative_cursor"
    return mode


def compute_cumulative_schedule(
    durations_sec: Iterable[float],
    fps: int = FPS_DEFAULT,
    is_continuations: Optional[Iterable[bool]] = None,
    context_frames: int = 22,
    av_sync_lead_sec: float = 0.0,
    duration_mode: str = "cumulative_cursor",
    overlap_overrides: Optional[Iterable[int]] = None,
) -> list[dict]:
    """Compute exact per-shot `keep_frames`, `trim_head_frames`, `target_frames`, gear fields, and bounds."""
    durs = [max(0.04, float(d)) for d in durations_sec]
    if is_continuations is None:
        conts = [i > 0 for i in range(len(durs))]
    else:
        conts = [bool(c) for c in is_continuations]
    if len(conts) != len(durs):
        raise ValueError("ShotVault: is_continuations length must match durations_sec.")

    base_ov = int(context_frames)
    if base_ov not in (22, 39, 56):
        raise ValueError(f"ShotVault: context_frames must be 22, 39, or 56, got {context_frames}.")

    overrides = [int(x or 0) for x in overlap_overrides] if overlap_overrides is not None else [0] * len(durs)
    if len(overrides) != len(durs):
        raise ValueError("ShotVault: overlap_overrides length must match durations_sec.")

    fps = max(1, int(fps))
    lead_sec = float(av_sync_lead_sec)
    lead_frames = round(lead_sec * fps)

    schedule: list[dict] = []
    cum_sec = 0.0
    cum_cursor = 0
    for idx, (dur, is_cont, ov_ovr) in enumerate(zip(durs, conts, overrides)):
        t_start = cum_sec
        t_end = cum_sec + dur
        cum_sec = t_end

        if ov_ovr in (22, 39, 56):
            shot_gear = ov_ovr
        else:
            shot_gear = base_ov

        if not is_cont:
            gear = compute_overlap_gear(0, base_gear=shot_gear, is_continuation=False)
            trim_head = max(0, lead_frames)
            overlap_frames = 0
            preroll_sec = max(0.0, trim_head / float(fps))
        else:
            ov = shot_gear
            if lead_frames > ov:
                ov = 39 if lead_frames <= 39 else 56
            gear = compute_overlap_gear(ov, base_gear=shot_gear, is_continuation=True)
            trim_head = gear["trim_head_frames"]
            overlap_frames = gear["overlap_frames"]
            preroll_sec = max(0.0, lead_sec)

        if duration_mode == "cumulative_cursor":
            target_cursor = max(cum_cursor + 1, round(t_end * fps))
            keep_frames = target_cursor - cum_cursor
            cum_start = cum_cursor
            cum_cursor = target_cursor
            target_frames = align_frame_count(keep_frames + trim_head)
        elif duration_mode == "net_preserve":
            keep_frames = max(1, round(dur * fps))
            cum_start = cum_cursor
            cum_cursor += keep_frames
            target_frames = align_frame_count(keep_frames + trim_head)
        elif duration_mode == "zero_tail_trim":
            raw_req = max(22 if is_cont else 5, round(dur * fps))
            target_frames = align_frame_count(raw_req + trim_head)
            keep_frames = max(1, target_frames - trim_head)
            cum_start = cum_cursor
            cum_cursor += keep_frames
        else:
            raise ValueError(f"ShotVault: unknown duration_mode={duration_mode!r}")
        schedule.append({
            "shot_index": idx + 1,
            "duration_sec": dur,
            "t_start_sec": round(t_start, 6),
            "t_end_sec": round(t_end, 6),
            "cum_frame_start": cum_start,
            "cum_frame_end": cum_cursor,
            "keep_frames": int(keep_frames),
            "trim_head_frames": int(trim_head),
            "overlap_frames": int(overlap_frames),
            "target_frames": int(target_frames),
            "preroll_sec": round(preroll_sec, 6),
            "is_continuation": bool(is_cont),
            "overlap_gear": gear["overlap_gear"],
            "taper_tail_frames": gear["taper_tail_frames"],
            "taper_ramp_frames": gear["taper_ramp_frames"],
            "audio_guide_frames": gear["audio_guide_frames"],
            "audio_feather_ticks": gear["audio_feather_ticks"],
            "continuation_pair": gear["continuation_pair"],
            "video_latent_steps": gear["video_latent_steps"],
        })
    return schedule


def compute_tail_bridge_plan(
    trim_head: int,
    keep_frames: int,
    tail_bridge_frames: int,
    audio_drive_active: bool = False,
) -> dict:
    """Compute H3 `17k+5`-aligned `(target_frames, keep_frames, tail_anchor_index, tail_bridge_frames)`.

    - `tail_bridge_frames == 0`: exact passthrough of original schedule.
    - Regular mode (`audio_drive_active == False`): places tail anchor flush at `target_frames - G`
      so `5f`/`22f` align strictly on `cycle_pos == 0` (`tail_idx % 17 == 0`) and `1f` matches official
      FL2V `target_frames - 1`, choosing the `17k+5` target closest to original `keep_frames`.
    - Audio-driven / MV mode (`audio_drive_active == True`): preserves bit-exact `keep_frames` (zero beat drift),
      placing tail anchor at `trim_head + keep_frames` and expanding `target_frames = align_frame_count(tail_idx + G)`.
    """
    th = max(0, int(trim_head))
    kf = max(1, int(keep_frames))
    g = int(tail_bridge_frames or 0)
    if g not in (1, 5, 22):
        base_tf = align_frame_count(th + kf)
        return {
            "target_frames": base_tf,
            "keep_frames": kf,
            "tail_anchor_index": -1,
            "tail_bridge_frames": 0,
        }

    if audio_drive_active:
        tail_idx = th + kf
        tf = align_frame_count(tail_idx + g)
        return {
            "target_frames": int(tf),
            "keep_frames": int(kf),
            "tail_anchor_index": int(tail_idx),
            "tail_bridge_frames": int(g),
        }

    # Regular mode: flush at clip end (target_frames - g), keeping minimum deviation from orig keep_frames
    min_tf = align_frame_count(th + max(17, kf // 2) + g)
    t_lo = max(min_tf, align_frame_count(th + kf))
    if t_lo - g <= th + 16:
        t_lo = align_frame_count(th + 17 + g)
    t_hi = t_lo + GROUP_FRAMES
    keep_lo = t_lo - g - th
    keep_hi = t_hi - g - th
    if abs(keep_lo - kf) <= abs(keep_hi - kf):
        tf, eff_keep = t_lo, keep_lo
    else:
        tf, eff_keep = t_hi, keep_hi
    tail_idx = tf - g
    return {
        "target_frames": int(tf),
        "keep_frames": int(eff_keep),
        "tail_anchor_index": int(tail_idx),
        "tail_bridge_frames": int(g),
    }
