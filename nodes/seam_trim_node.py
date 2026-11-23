"""[N4] ShotVaultSeamTrim: Symmetric AV trim, auto ΔE dual-layer Lab MKL lock, and zero-delta RIFE bridge."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..compat import resample_audio
from ..core.grid_math import DELIVERY_AUDIO_SR
from ..core.rife_micro_bridge import apply_zero_delta_rife_micro_bridge
from ..core.seam_and_color import (
    auto_dual_mkl_lab_match,
    detect_seam_gate_and_drift,
    heal_single_video_o1,
    linear_22f_mkl_lab_match,
    masked_bg_ycc_match,
    mkl_lab_seam_match,
    trim_av_sync,
)


class ShotVaultSeamTrim:
    """[N4] Port-0 aligned `IMAGE -> IMAGE` operator trimming overlap/tail, locking color, and RIFE-bridging jumps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_head_frames": ("INT", {"default": -1, "min": -1, "max": 4096}),
                "keep_frames": ("INT", {"default": -1, "min": -1, "max": 36000}),
                "color_match_mode": (
                    [
                        "linear_22f_mkl_lab",
                        "auto_dual_mkl_lab",
                        "seam_12f_mkl_lab",
                        "masked_bg_ycc",
                        "single_video_auto_heal",
                        "off",
                    ],
                    {"default": "linear_22f_mkl_lab"},
                ),
                "seam_frames": ("INT", {"default": 22, "min": 1, "max": 64}),
                "strength_curve": ("STRING", {"default": "0.88"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "reference_clean_tail": ("IMAGE",),
                "source_bg_frames": ("IMAGE",),
                "shot_pipe": ("SHOT_PIPE",),
                "rife_bridge_mode": (["auto_jump_rife", "always_rife", "off"], {"default": "off"}),
                "bridge_max_frames": ("INT", {"default": 4, "min": 1, "max": 8}),
                "color_deadzone_de": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "global_lock_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "jump_spike_threshold": ("FLOAT", {"default": 1.65, "min": 1.0, "max": 5.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim_and_match"
    CATEGORY = "Acheng-ShotVault/Signal"

    def trim_and_match(
        self,
        images,
        trim_head_frames: int,
        keep_frames: int,
        color_match_mode: str,
        seam_frames: int,
        strength_curve: str,
        audio=None,
        reference_clean_tail=None,
        source_bg_frames=None,
        shot_pipe: Optional[dict] = None,
        rife_bridge_mode: str = "off",
        bridge_max_frames: int = 4,
        color_deadzone_de: float = 1.5,
        global_lock_strength: float = 0.75,
        jump_spike_threshold: float = 1.65,
    ):
        pipe = shot_pipe if isinstance(shot_pipe, dict) else None
        p_dict = pipe or {}
        fps = int(p_dict.get("fps", 24))
        eff_trim = int(p_dict.get("trim_head_frames", 0)) if int(trim_head_frames) < 0 else int(trim_head_frames)
        eff_keep = int(p_dict.get("keep_frames", 0)) if int(keep_frames) < 0 else int(keep_frames)

        trimmed_img, trimmed_aud = trim_av_sync(
            images,
            audio,
            trim_head_frames=eff_trim,
            keep_frames=eff_keep,
            fps=fps,
        )

        is_cont = True if pipe is None else bool(p_dict.get("is_continuation", True))
        known_cont = bool(pipe is not None and is_cont)
        ref_tail = reference_clean_tail if reference_clean_tail is not None else p_dict.get("parent_clean_tail_ref")
        has_ref = ref_tail is not None and float(ref_tail.abs().max().item()) > 0.0
        scene_anchor = p_dict.get("scene_anchor_clean_ref")

        color_report: dict = {"mode": color_match_mode, "applied_color": False}
        rife_report: dict = {"mode": rife_bridge_mode, "applied_rife": False, "rife_frames": 0}
        gate_info: Optional[dict] = None

        if color_match_mode == "single_video_auto_heal":
            trimmed_img, trimmed_aud, heal_rep = heal_single_video_o1(
                trimmed_img,
                audio=trimmed_aud,
                fps=fps,
                seam_frames=int(seam_frames),
                bridge_max_frames=int(bridge_max_frames),
                deadzone_de=float(color_deadzone_de),
                global_lock_strength=float(global_lock_strength),
                jump_spike_threshold=float(jump_spike_threshold),
            )
            color_report = heal_rep
        elif color_match_mode == "linear_22f_mkl_lab" and is_cont and has_ref:
            trimmed_img, lin_rep = linear_22f_mkl_lab_match(
                trimmed_img,
                ref_tail,
                seam_frames=int(seam_frames),
                initial_strength=strength_curve,
            )
            color_report = lin_rep
        elif color_match_mode == "auto_dual_mkl_lab" and is_cont and has_ref:
            trimmed_img, gate_info = auto_dual_mkl_lab_match(
                trimmed_img,
                ref_tail,
                scene_anchor_clean_ref=scene_anchor,
                seam_frames=int(seam_frames),
                deadzone_de=float(color_deadzone_de),
                global_lock_strength=float(global_lock_strength),
                known_continuation=known_cont,
            )
            color_report = dict(gate_info, mode=color_match_mode)
        elif color_match_mode == "seam_12f_mkl_lab" and is_cont and has_ref:
            trimmed_img = mkl_lab_seam_match(
                trimmed_img,
                ref_tail,
                seam_frames=int(seam_frames),
                strength_curve=strength_curve,
            )
            color_report = {"mode": color_match_mode, "applied_color": True}
        elif color_match_mode == "masked_bg_ycc" and source_bg_frames is not None:
            trimmed_img, bg_rep = masked_bg_ycc_match(trimmed_img, source_bg_frames)
            color_report = {"mode": color_match_mode, "shots": bg_rep}

        if (
            color_match_mode != "single_video_auto_heal"
            and is_cont
            and has_ref
            and str(rife_bridge_mode or "auto_jump_rife") != "off"
        ):
            if gate_info is None:
                gate_info = detect_seam_gate_and_drift(
                    trimmed_img,
                    ref_tail,
                    scene_anchor=scene_anchor,
                    deadzone_de=float(color_deadzone_de),
                    known_continuation=known_cont,
                )
            trimmed_img, rife_report = apply_zero_delta_rife_micro_bridge(
                trimmed_img,
                ref_tail,
                mode=str(rife_bridge_mode),
                max_bridge_frames=int(bridge_max_frames),
                jump_spike_threshold=float(jump_spike_threshold),
                gate_info=gate_info,
                known_continuation=known_cont,
            )

        if pipe is not None:
            pipe["last_seam_trim_report"] = {"color": color_report, "rife": rife_report}

        if pipe and (trimmed_aud is None or bool(p_dict.get("audio_drive_active"))):
            ref = p_dict.get("delivery_ref_audio")
            if isinstance(ref, dict) and isinstance(ref.get("waveform"), torch.Tensor) and ref["waveform"].numel() > 0:
                w = ref["waveform"].detach().cpu().float()
                if w.ndim == 3:
                    w = w[0]
                elif w.ndim == 1:
                    w = w.unsqueeze(0)
                if w.shape[0] == 1:
                    w = w.repeat(2, 1)
                elif w.shape[0] > 2:
                    w = w[:2]
                w48 = resample_audio(w, int(ref.get("sample_rate", 48000)), DELIVERY_AUDIO_SR)
                need = max(1, round(int(trimmed_img.shape[0]) / float(max(1, fps)) * DELIVERY_AUDIO_SR))
                if w48.shape[1] < need:
                    w48 = F.pad(w48, (0, need - w48.shape[1]))
                else:
                    w48 = w48[:, :need]
                trimmed_aud = {"waveform": w48.unsqueeze(0).contiguous(), "sample_rate": DELIVERY_AUDIO_SR}

        return (trimmed_img, trimmed_aud)
