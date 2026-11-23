"""Phase 9 deterministic regression tests:
1. 22-frame dense arithmetic linear MKL-Lab color match (`prev[-1]` sole ref, constant step delta, VFX preservation, bad-head guard).
2. 1114 continuation audio guide (`22f->20f`, `39f->37f`, `56f->54f` 2-frame anti-pop guard band + WindowGuider priority).
3. Audio-driven 39-frame overlap conservation (`+39f overlap -> -39f trim == exact sum of segment durations`, 0 sample drift).
4. Upgraded AchengSeamBridge (`seam_bridge.py`): zero false white-head misfires, adaptive bridge span, head freeze+snap & delayed token jump repair, and hard-cut protection.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import torch

_CUSTOM_NODES_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CUSTOM_NODES_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_NODES_DIR)

_PKG_NAME = "ComfyUI_Universal_ShotVault_TestPkg"
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        os.path.join(_PKG_DIR, "__init__.py"),
        submodule_search_locations=[_PKG_DIR],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = mod
    spec.loader.exec_module(mod)

from ComfyUI_Universal_ShotVault_TestPkg.core.asset_and_turnaround import (  # noqa: E402
    build_continuation_driving_audio,
    slice_1114_prev_audio_guide,
)
from ComfyUI_Universal_ShotVault_TestPkg.core.context_window_guider import build_window_guider_and_latent  # noqa: E402
from ComfyUI_Universal_ShotVault_TestPkg.core.grid_math import (  # noqa: E402
    audio_latent_t,
    compute_cumulative_schedule,
    resolve_duration_mode,
    video_latent_t,
)
from ComfyUI_Universal_ShotVault_TestPkg.core.seam_and_color import (  # noqa: E402
    _rgb_to_lab_exact,
    linear_22f_mkl_lab_match,
    trim_av_sync,
)

# Load AchengSeamBridge directly from sibling custom node package
_SB_PATH = os.path.join(_CUSTOM_NODES_DIR, "ComfyUI-Acheng-SeamBridge", "seam_bridge.py")
_sb_spec = importlib.util.spec_from_file_location("acheng_seam_bridge_mod", _SB_PATH)
_sb_mod = importlib.util.module_from_spec(_sb_spec)
_sb_spec.loader.exec_module(_sb_mod)
AchengSeamBridge = _sb_mod.AchengSeamBridge


class _MockVideoVAE:
    def encode(self, img_bhwc: torch.Tensor) -> torch.Tensor:
        f = int(img_bhwc.shape[0])
        h, w = int(img_bhwc.shape[1]) // 16, int(img_bhwc.shape[2]) // 16
        return torch.ones((1, 24, video_latent_t(f), max(1, h), max(1, w)), dtype=torch.float32) * 0.42


class _MockAudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.last_input_samples = 0

    def encode(self, wav_blc: torch.Tensor) -> torch.Tensor:
        # wav_blc is [1, L, C] matching ComfyUI AudioVAE contract
        length = int(wav_blc.shape[1])
        self.last_input_samples = length
        tokens = max(1, round(length / 800.0))
        return torch.ones((1, 32, 2, tokens), dtype=torch.float32) * 0.33


class TestPhase9ContinuationAndBridge(unittest.TestCase):
    def test_linear_22f_mkl_lab_sole_prev_last_and_constant_step(self):
        torch.manual_seed(7)
        h, w = 64, 96
        yy = torch.linspace(0.15, 0.75, h).view(1, h, 1, 1).expand(8, h, w, 3)
        xx = torch.linspace(0.10, 0.80, w).view(1, 1, w, 1).expand(8, h, w, 3)
        base_frame = (0.55 * yy + 0.45 * xx).clamp(0.05, 0.90)

        # Previous tail: make frames [0..6] wildly purple, ONLY prev[-1] warm forest tone
        prev_tail_a = base_frame.clone()
        prev_tail_a[:-1, ..., 0] = 0.85
        prev_tail_a[:-1, ..., 1] = 0.10
        prev_tail_a[:-1, ..., 2] = 0.85
        prev_tail_a[-1, ..., 0] = (base_frame[-1, ..., 0] * 0.90 + 0.04).clamp(0, 1)
        prev_tail_a[-1, ..., 1] = (base_frame[-1, ..., 1] * 1.08 + 0.05).clamp(0, 1)
        prev_tail_a[-1, ..., 2] = (base_frame[-1, ..., 2] * 0.82).clamp(0, 1)

        prev_tail_b = prev_tail_a.clone()
        prev_tail_b[:-1] = 0.02  # Completely different non-final frames

        # Current 40-frame segment with cold blue-dark LoRA drift + emerald VFX flash on frame 10
        curr = torch.empty((40, h, w, 3), dtype=torch.float32)
        for i in range(40):
            curr[i, ..., 0] = (base_frame[0, ..., 0] * 0.72).clamp(0, 1)
            curr[i, ..., 1] = (base_frame[0, ..., 1] * 0.78).clamp(0, 1)
            curr[i, ..., 2] = (base_frame[0, ..., 2] * 1.12 + 0.06).clamp(0, 1)
        curr_vfx = curr.clone()
        curr_vfx[10, 16:48, 24:72, 1] = 0.96  # Bright emerald energy arrow burst at frame 10

        out_a, rep_a = linear_22f_mkl_lab_match(curr, prev_tail_a, seam_frames=22, initial_strength=0.88)
        out_b, _ = linear_22f_mkl_lab_match(curr, prev_tail_b, seam_frames=22, initial_strength=0.88)

        # 1. Strictly anchored ONLY to prev[-1]
        self.assertEqual(float((out_a - out_b).abs().max().item()), 0.0)
        # 2. Frames 22..39 are 100% untouched bit-for-bit
        self.assertEqual(float((out_a[22:] - curr[22:]).abs().max().item()), 0.0)
        # 3. Seam ΔE drops dramatically on frame 0
        self.assertLess(rep_a["delta_e_after"], rep_a["delta_e_seam"] * 0.22)
        self.assertAlmostEqual(rep_a["step_delta"], 0.88 / 22.0, places=5)

        # 4. Verify strict linear Lab progression across frames 0..22 (constant per-frame Lab step!)
        labs = _rgb_to_lab_exact(out_a[:23]).mean(dim=(2, 3))  # [23, 3]
        step_norms = torch.norm(labs[1:] - labs[:-1], dim=-1)  # 22 steps: 0->1, ..., 21->22
        self.assertLess(float((step_norms.max() - step_norms.min()).item()), 0.25)

        # 5. Intra-shot VFX burst on frame 10 stays bright green (not crushed!)
        out_vfx, _ = linear_22f_mkl_lab_match(curr_vfx, prev_tail_a, seam_frames=22, initial_strength=0.88)
        self.assertGreater(float(out_vfx[10, 16:48, 24:72, 1].mean().item()), 0.85)

    def test_1114_audio_guide_2f_guard_band_and_window_guider_gears(self):
        import comfy.nested_tensor  # type: ignore

        raw_tail = torch.linspace(0.05, 0.95, 96000).unsqueeze(0).repeat(2, 1)  # 3.0s @ 32kHz
        for ov, exp_ag, exp_feather in ((22, 20, 7), (39, 37, 12), (56, 54, 16)):
            sliced = slice_1114_prev_audio_guide(raw_tail, overlap_frames=ov, audio_guide_frames=exp_ag, fps=24)
            wav = sliced["waveform"]
            exp_samples = round(exp_ag / 24.0 * 32000)
            ov_samples = round(ov / 24.0 * 32000)
            self.assertEqual(int(wav.shape[-1]), exp_samples)
            # Confirm it starts at -ov_samples and drops the final 2 frames (~2667 samples)
            self.assertTrue(torch.allclose(wav[0, 0, 0], raw_tail[0, -ov_samples], atol=1e-6))
            self.assertLess(float(wav[0, 0, -1].item()), float(raw_tail[0, -1].item()) - 0.01)

            # Confirm WindowGuider uses 1114 prev_audio (NOT un-shortened prev_tail_latent) when tapered_prev_images is wired
            v_vae = _MockVideoVAE()
            a_vae = _MockAudioVAE()
            total_f = 226
            v_z = torch.zeros((1, 24, video_latent_t(total_f), 8, 8))
            a_z = torch.zeros((1, 32, 2, audio_latent_t(total_f)))
            lat = {"samples": comfy.nested_tensor.NestedTensor((v_z, a_z))}
            pos = [[torch.zeros((1, 16, 5120)), {}]]
            # Deliberately supply poison 9.99 prev_tail_latent audio to prove 1114 prev_audio wins!
            poison_prev_lat = {
                "samples": comfy.nested_tensor.NestedTensor((
                    torch.ones((1, 24, video_latent_t(ov), 8, 8)) * 9.99,
                    torch.ones((1, 32, 2, audio_latent_t(ov))) * 9.99,
                )),
                "shotvault_meta": {"audio_overhang": 0.0},
            }
            pipe = {
                "current_shot_index": 3,
                "parent_take_id": "shot_0002_take_001",
                "is_continuation": True,
                "overlap_frames": ov,
                "audio_guide_frames": exp_ag,
                "gear_feather_ticks": exp_feather,
                "auto_scale_feather": True,
            }
            tapered = torch.rand((ov, 128, 128, 3))
            cond_out, _ = build_window_guider_and_latent(
                model=None,
                positive=pos,
                latent=lat,
                vae=v_vae,
                audio_vae=a_vae,
                tapered_prev_images=tapered,
                prev_audio=sliced,
                prev_tail_latent=poison_prev_lat,
                shot_pipe=pipe,
                pass_stage="single_or_1st",
                continuation_frames="auto",
                keyframe_until_sigma=0.0,
                audio_feather_ticks=7,
            )
            kfs = cond_out[0][1]["minimax_keyframes"]
            self.assertEqual(len(kfs), 1)  # Combined video+audio dict at resolved_frame_index=0
            self.assertEqual(kfs[0]["resolved_frame_index"], 0)
            self.assertEqual(kfs[0]["audio_feather_ticks"], exp_feather)
            self.assertAlmostEqual(float(kfs[0]["audio_latent"].mean().item()), 0.33, places=4)
            self.assertEqual(int(kfs[0]["audio_latent"].shape[-1]), audio_latent_t(exp_ag))

    def test_audio_drive_39f_overlap_exact_sum_conservation(self):
        # Fractional segment durations: 3.11s + 4.37s + 5.52s = 13.00s -> round(13.0 * 24) = 312 exact frames!
        durs = [3.11, 4.37, 5.52]
        sched = compute_cumulative_schedule(
            durs,
            fps=24,
            is_continuations=[False, True, True],
            context_frames=39,
            duration_mode=resolve_duration_mode("zero_tail_trim", audio_drive_active=True),
            overlap_overrides=[0, 39, 39],
        )
        self.assertEqual(sched[0]["overlap_frames"], 0)
        self.assertEqual(sched[1]["overlap_frames"], 39)
        self.assertEqual(sched[2]["overlap_frames"], 39)
        self.assertEqual(sched[1]["trim_head_frames"], 39)
        self.assertEqual(sched[2]["trim_head_frames"], 39)

        # Verify each continuation generates align_17k+5(keep + 39) and trims back to exact keep_frames
        delivered_counts = []
        delivered_48k_samples = 0
        sr = 48000
        master_wav = torch.sin(torch.linspace(0, 200.0, int(15 * sr))).unsqueeze(0).repeat(2, 1)
        per_shot_map = {}
        for i, sc in enumerate(sched, 1):
            s0 = round((sc["cum_frame_start"] / 24.0) * sr)
            s1 = round((sc["cum_frame_end"] / 24.0) * sr)
            per_shot_map[i] = {"waveform": master_wav[:, s0:s1].unsqueeze(0), "sample_rate": sr}

        parent_ctx = None
        for i, sc in enumerate(sched, 1):
            guide_aud, deliv_ref = build_continuation_driving_audio(
                per_shot_map[i],
                curr_idx=i,
                sched=sc,
                fps=24,
                parent_ctx=parent_ctx,
                per_shot_audio_map=per_shot_map,
            )
            if i > 1:
                # 39 frames @ 32kHz = exact 52,000 prepended samples before current net audio!
                min_expected_32k = 52000 + round(sc["keep_frames"] / 24.0 * 32000)
                self.assertGreaterEqual(int(guide_aud["waveform"].shape[-1]), min_expected_32k)
            raw_gen = torch.zeros((sc["target_frames"], 32, 32, 3))
            trimmed_img, trimmed_aud = trim_av_sync(
                raw_gen,
                deliv_ref,
                trim_head_frames=sc["trim_head_frames"],
                keep_frames=sc["keep_frames"],
                fps=24,
            )
            delivered_counts.append(int(trimmed_img.shape[0]))
            delivered_48k_samples += int(deliv_ref["waveform"].shape[-1])
            parent_ctx = {"tensors": {"audio_32k_tail": torch.ones((2, 76800)) * 0.2}}

        self.assertEqual(sum(delivered_counts), round(sum(durs) * 24))
        self.assertEqual(delivered_48k_samples, sum(delivered_counts) * 2000)  # 0-sample drift @ 48kHz!

    def test_acheng_seam_bridge_no_false_white_and_repairs_mutations(self):
        bridge = AchengSeamBridge()
        h, w = 96, 128
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        scene = (
            0.35
            + 0.25 * torch.exp(-((xx - 0.45) ** 2 + (yy - 0.50) ** 2) / 0.03)
            + 0.18 * torch.sin(xx * 11.0 + yy * 7.0)
            + 0.12 * torch.cos(xx * 23.0 - yy * 17.0)
        ).unsqueeze(-1).repeat(1, 1, 3).clamp(0.0, 1.0)

        def shift_frame(dx_px: float) -> torch.Tensor:
            shift = int(round(dx_px))
            return torch.roll(scene, shifts=shift, dims=1).clamp(0.0, 1.0)

        # 1. Normal camera brightness tilt (+0.10 brighter at frame 0 vs frame 10, zero highlight clipping) -> NO false white trigger!
        prev_tail = torch.stack([shift_frame(k) * 0.95 for k in range(-7, 1)], dim=0)
        curr_tilt = torch.stack([(shift_frame(k + 1) * (1.0 - 0.012 * min(k, 10))).clamp(0, 0.88) for k in range(16)], dim=0)
        out_tilt, _, diag_tilt_s = bridge.repair(
            curr_tilt, mode="auto", previous_tail=prev_tail, is_continuation=True,
            auto_bridge_backend="flow", auto_align_color=False,
        )
        diag_tilt = json.loads(diag_tilt_s)
        self.assertEqual(diag_tilt.get("white_head_frames", 0), 0)
        self.assertEqual(diag_tilt["reason"], "smooth_passthrough")
        self.assertEqual(float((out_tilt - curr_tilt).abs().max().item()), 0.0)

        # 2. H3 Head Micro-Freeze (2 frozen frames at curr[0..1] == prev[-1], then sudden snap at curr[2])
        prev_mov = torch.stack([shift_frame(k * 2) for k in range(-7, 1)], dim=0)
        curr_freeze = [shift_frame(0), shift_frame(0)] + [shift_frame((k + 1) * 2) for k in range(2, 16)]
        curr_freeze_t = torch.stack(curr_freeze, dim=0)
        out_frz, _, diag_frz_s = bridge.repair(
            curr_freeze_t, mode="auto", previous_tail=prev_mov, is_continuation=True,
            auto_bridge_backend="flow", auto_align_color=False,
        )
        diag_frz = json.loads(diag_frz_s)
        self.assertEqual(diag_frz["mutation_kind"], "head_freeze_snap")
        self.assertEqual(diag_frz["reason"], "repaired")
        self.assertEqual(int(out_frz.shape[0]), int(curr_freeze_t.shape[0]))
        self.assertLess(diag_frz["repaired_peak_mae"], diag_frz["raw_peak_mae"])

        # 3. Delayed Causal-Token Overhang Jump at curr[3] (curr[0..2] smooth, sudden 8px jump at curr[2]->curr[3])
        curr_delay = [shift_frame(2), shift_frame(4), shift_frame(6)] + [shift_frame(6 + 8 + (k - 3) * 2) for k in range(3, 16)]
        curr_delay_t = torch.stack(curr_delay, dim=0)
        out_del, _, diag_del_s = bridge.repair(
            curr_delay_t, mode="auto", previous_tail=prev_mov, is_continuation=True,
            auto_bridge_backend="flow", auto_align_color=False,
        )
        diag_del = json.loads(diag_del_s)
        self.assertEqual(diag_del["mutation_kind"], "delayed_token_jump")
        self.assertEqual(diag_del["reason"], "repaired")
        self.assertLessEqual(diag_del["repair_offset"], 3)
        self.assertGreaterEqual(diag_del["b_endpoint_frame"], 3)
        self.assertLess(diag_del["repaired_peak_mae"], diag_del["raw_peak_mae"])

        # 4. True Hard Cut (completely unrelated pattern) -> safely protected, 0 frames modified!
        unrelated = ((torch.cos(yy * 45.0) > 0).float() * 0.8 + 0.1).unsqueeze(-1).repeat(16, 1, 1, 3)
        out_cut, _, diag_cut_s = bridge.repair(
            unrelated, mode="auto", previous_tail=prev_mov, is_continuation=True,
            auto_bridge_backend="flow", auto_align_color=False,
        )
        diag_cut = json.loads(diag_cut_s)
        self.assertIn("hard_cut", diag_cut["reason"])
        self.assertEqual(diag_cut["changed_frames"], 0)

    def test_acheng_seam_bridge_unconditional_rife_no_skip_no_revert(self):
        bridge = AchengSeamBridge()
        h, w = 96, 128
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        scene = (
            0.35
            + 0.25 * torch.exp(-((xx - 0.45) ** 2 + (yy - 0.50) ** 2) / 0.03)
            + 0.18 * torch.sin(xx * 11.0 + yy * 7.0)
            + 0.12 * torch.cos(xx * 23.0 - yy * 17.0)
        ).unsqueeze(-1).repeat(1, 1, 3).clamp(0.0, 1.0)

        def shift_frame(dx_px: float) -> torch.Tensor:
            return torch.roll(scene, shifts=int(round(dx_px)), dims=1).clamp(0.0, 1.0)

        # Already smooth continuation: old `auto` skipped it, but unconditional `rife` bridges exact 3 frames!
        prev_smooth = torch.stack([shift_frame(k) for k in range(-7, 1)], dim=0)
        curr_smooth = torch.stack([shift_frame(k + 1) for k in range(16)], dim=0)

        out_b, _, diag_s = bridge.repair(
            curr_smooth,
            mode="rife",
            join_frame=0,
            max_bridge_frames=3,
            release_frames=12,
            sensitivity_ratio=1.65,
            protect_hard_cuts=False,
            repair_white_head=False,
            previous_tail=prev_smooth,
            is_continuation=True,
            auto_bridge_backend="rife",
            rife_checkpoint="rife49.pth",
            auto_align_color=False,
        )
        diag = json.loads(diag_s)
        self.assertEqual(diag["reason"], "repaired")
        self.assertEqual(diag["mutation_kind"], "unconditional_bridge")
        self.assertEqual(diag["changed_frames"], 3)
        self.assertEqual(diag["frame_delta"], 0)
        self.assertEqual(diag["b_endpoint_frame"], 3)
        self.assertEqual(int(out_b.shape[0]), int(curr_smooth.shape[0]))
        self.assertEqual(float((out_b[3:] - curr_smooth[3:]).abs().max().item()), 0.0)
        # Standalone unittest has no full ComfyUI `nodes` runtime, so RIFE gracefully falls back to opencv_dis
        self.assertTrue(diag["backend"].startswith("rife:") or diag["backend"] == "opencv_dis")

        # On Shot 1 or explicit Director hard cut (is_continuation=False), cleanly skip with 0 changes
        out_cut, _, diag_cut_s = bridge.repair(
            curr_smooth,
            mode="rife",
            join_frame=0,
            max_bridge_frames=3,
            protect_hard_cuts=False,
            repair_white_head=False,
            previous_tail=prev_smooth,
            is_continuation=False,
            auto_bridge_backend="rife",
            rife_checkpoint="rife49.pth",
            auto_align_color=False,
        )
        diag_cut = json.loads(diag_cut_s)
        self.assertEqual(diag_cut["reason"], "disabled_or_not_continuation")
        self.assertEqual(diag_cut["changed_frames"], 0)
        self.assertEqual(float((out_cut - curr_smooth).abs().max().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
