"""Phase 8 tests: v5.3 3-gate auto ΔE dual-layer MKL lock, FP16 RIFE 0-delta micro-bridge, O(1) heal & flagship aesthetic workflow."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_NODES_DIR = os.path.dirname(PKG_DIR)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES_DIR)
for p in (COMFY_ROOT, CUSTOM_NODES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

sv_pkg = importlib.import_module("ComfyUI-Acheng-ShotVault")
sc_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.seam_and_color")
rb_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.rife_micro_bridge")
vs_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.vault_store")


def _make_textured_clip(
    n_frames: int,
    h: int = 128,
    w: int = 128,
    shift_x: float = 0.0,
    vx: float = 0.0,
    color_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij")
    frames = []
    for t in range(n_frames):
        dx = (shift_x + vx * t) * (2.0 / float(w))
        x = torch.remainder(xx - dx + 1.0, 2.0) - 1.0
        base = 0.42 + 0.16 * torch.sin(x * 16.0) * torch.cos(yy * 14.0) + 0.10 * torch.cos(x * 29.0 + yy * 19.0)
        box = ((x.abs() < 0.28) & (yy.abs() < 0.35)).float() * 0.18
        r = (base + box * 0.6 + color_bias[0]).clamp(0.02, 0.98)
        g = (base + box * 0.2 + color_bias[1]).clamp(0.02, 0.98)
        b = (base - box * 0.3 + color_bias[2]).clamp(0.02, 0.98)
        frames.append(torch.stack([r, g, b], dim=-1))
    return torch.stack(frames, dim=0).float()


class TestPhase8AutoColorLockRifeBridgeAndAestheticWf(unittest.TestCase):
    def test_3gate_deadzone_hardcut_and_flash_immunity(self):
        ref = _make_textured_clip(12, shift_x=0.0, vx=0.5)

        # Case A: tiny drift below JND deadzone -> zero pixel modification
        curr_ok = _make_textured_clip(16, shift_x=6.0, vx=0.5, color_bias=(0.002, 0.0, -0.002))
        out_ok, gate_ok = sc_mod.auto_dual_mkl_lab_match(curr_ok, ref[-4:], scene_anchor_clean_ref=ref[:4], deadzone_de=1.5)
        self.assertTrue(gate_ok["same_scene_safe"])
        self.assertFalse(gate_ok["applied_color"])
        self.assertEqual(float((out_ok - curr_ok).abs().max().item()), 0.0)

        # Case B: unrelated hard cut -> safe gate False -> zero pixel modification
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 128), torch.linspace(0, 1, 128), indexing="ij")
        cut_base = torch.stack([
            0.5 + 0.45 * torch.sin(yy * 40.0 + xx * 9.0),
            0.2 + 0.45 * torch.cos(xx * 37.0 - yy * 13.0),
            0.7 - 0.35 * torch.sin((xx - yy) * 28.0),
        ], dim=-1).clamp(0.02, 0.98).unsqueeze(0).repeat(16, 1, 1, 1)
        out_cut, gate_cut = sc_mod.auto_dual_mkl_lab_match(cut_base, ref[-4:])
        self.assertFalse(gate_cut["same_scene_safe"])
        self.assertFalse(gate_cut["applied_color"])
        self.assertEqual(float((out_cut - cut_base).abs().max().item()), 0.0)

        # Case C: 2-frame martial-arts lightning flash snapping back to reference -> protected
        curr_flash = _make_textured_clip(16, shift_x=6.0, vx=0.5)
        curr_flash[:2] = (curr_flash[:2] + 0.42).clamp(0.0, 1.0)
        out_fl, gate_fl = sc_mod.auto_dual_mkl_lab_match(curr_flash, ref[-4:])
        self.assertTrue(gate_fl["is_flash"])
        self.assertFalse(gate_fl["applied_color"])
        self.assertEqual(float((out_fl - curr_flash).abs().max().item()), 0.0)

    def test_dual_layer_mkl_lab_stops_5shot_cumulative_color_snowball(self):
        s1 = _make_textured_clip(24, color_bias=(0.0, 0.0, 0.0))
        ref_mean = s1[-1].mean(dim=(0, 1))

        # Classical 12f decay-only vs v5.3 dual-layer global lock + save_take archival across Shots 2..5
        with tempfile.TemporaryDirectory() as tmp:
            store = vs_mod.VaultStore(tmp)
            shots = [{"shot_index": i, "is_continuation": (i > 1)} for i in range(1, 6)]
            store.save_take_context("s1_t1", 1, s1, None, 48000, 24, "a", "", "1", 24, 0, 1)

            cur_classic = s1
            cur_dual = s1
            for idx in range(2, 6):
                # Relative drift added to incoming previous tail (warm drift +R, -B every hop)
                raw_next_c = (cur_classic[-1].mean(dim=(0, 1)) - ref_mean)
                raw_c = _make_textured_clip(24, color_bias=(float(raw_next_c[0]) + 0.06, 0.0, float(raw_next_c[2]) - 0.05))
                cur_classic = sc_mod.mkl_lab_seam_match(raw_c, cur_classic[-4:], seam_frames=12)

                parent_tid = store.get_active_take_id(idx - 1)
                anc = vs_mod.load_scene_anchor_clean_ref(store, shots, idx, True, parent_tid, cur_dual[-4:])
                raw_next_d = (cur_dual[-1].mean(dim=(0, 1)) - ref_mean)
                raw_d = _make_textured_clip(24, color_bias=(float(raw_next_d[0]) + 0.06, 0.0, float(raw_next_d[2]) - 0.05))
                cur_dual, gate = sc_mod.auto_dual_mkl_lab_match(raw_d, cur_dual[-4:], scene_anchor_clean_ref=anc, seam_frames=14, deadzone_de=1.5, global_lock_strength=0.75)
                self.assertTrue(gate["applied_color"])
                store.save_take_context(f"s{idx}_t1", idx, cur_dual, None, 48000, 24, "a", parent_tid, "1", 24, 0, 1)

            drift_classic = float((cur_classic[-1].mean(dim=(0, 1)) - ref_mean).abs().max().item())
            drift_dual = float((cur_dual[-1].mean(dim=(0, 1)) - ref_mean).abs().max().item())
            self.assertGreater(drift_classic, 0.18)
            self.assertLess(drift_dual, drift_classic * 0.35)
            self.assertLess(drift_dual, 0.06)

    def test_static_background_mask_ignores_foreground_fireball(self):
        ref = _make_textured_clip(8, color_bias=(-0.05, 0.0, 0.10))
        curr = _make_textured_clip(16, color_bias=(0.05, 0.01, 0.01))
        # Inject localized bright red fireball occupying ~30% center foreground
        curr[:, 28:100, 28:100, 0] = 0.96
        curr[:, 28:100, 28:100, 1] = 0.15
        curr[:, 28:100, 28:100, 2] = 0.05

        out, gate = sc_mod.auto_dual_mkl_lab_match(curr, ref[-4:], scene_anchor_clean_ref=ref[:4], deadzone_de=1.5)
        self.assertTrue(gate["same_scene_safe"])
        self.assertTrue(gate["applied_color"])
        # Corners must keep reference cool-blue balance and NOT be pulled warm-red by center fireball
        corner_ref_b = float(ref[-1, :20, :20, 2].mean().item())
        corner_out_b = float(out[-1, :20, :20, 2].mean().item())
        corner_out_r = float(out[-1, :20, :20, 0].mean().item())
        self.assertAlmostEqual(corner_out_b, corner_ref_b, delta=0.05)
        self.assertLess(corner_out_r, corner_out_b)

    def test_zero_delta_real_rife49_micro_bridge_on_pose_jump(self):
        self.assertIsNotNone(rb_mod.load_cached_rife())
        trim_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSeamTrim"]()

        # Smooth sequence -> must NOT trigger RIFE
        ref = _make_textured_clip(8, shift_x=0.0, vx=1.2)
        curr_smooth = _make_textured_clip(16, shift_x=8 * 1.2, vx=1.2)
        pipe_s = {"fps": 24, "trim_head_frames": 0, "keep_frames": 16, "is_continuation": True}
        out_s, _ = trim_n.trim_and_match(curr_smooth, -1, -1, "auto_dual_mkl_lab", 14, "0.75, 0.45, 0.18", reference_clean_tail=ref, shot_pipe=pipe_s, rife_bridge_mode="auto_jump_rife")
        self.assertFalse(pipe_s["last_seam_trim_report"]["rife"]["applied_rife"])
        self.assertEqual(float((out_s - curr_smooth).abs().max().item()), 0.0)

        # Sudden non-aliasing subject pose jump at seam -> real rife49.pth micro-bridge triggers with 0 frame delta!
        yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, 128), torch.linspace(-1.0, 1.0, 128), indexing="ij")
        bg_r = 0.25 + 0.12 * torch.sin(xx * 6.0) * torch.cos(yy * 5.0)
        bg_g = 0.28 + 0.10 * torch.cos(xx * 4.0 + yy * 6.0)
        bg_b = 0.34 + 0.10 * torch.sin((xx - yy) * 5.0)
        def _mk_subject_clip(n: int, x0: float, vx: float) -> torch.Tensor:
            seq = []
            for t in range(n):
                cx = (x0 + vx * t) * (2.0 / 128.0)
                d2 = ((xx - cx) ** 2 + (yy * 1.1) ** 2) * 18.0
                blob = torch.exp(-d2)
                seq.append(torch.stack([
                    (bg_r + 0.55 * blob).clamp(0.02, 0.98),
                    (bg_g + 0.35 * blob).clamp(0.02, 0.98),
                    (bg_b - 0.15 * blob).clamp(0.02, 0.98),
                ], dim=-1))
            return torch.stack(seq, dim=0).float()

        ref_sub = _mk_subject_clip(8, x0=-12.0, vx=1.0)
        curr_jump = _mk_subject_clip(16, x0=8.0, vx=1.0)  # +13px pose jump relative to ref_sub[-1]
        aud = {"waveform": torch.sin(torch.linspace(0, 20, round(16 / 24.0 * 48000))).repeat(1, 2, 1) * 0.2, "sample_rate": 48000}
        pipe_j = {"fps": 24, "trim_head_frames": 0, "keep_frames": 16, "is_continuation": True}
        out_j, out_aud = trim_n.trim_and_match(curr_jump, -1, -1, "auto_dual_mkl_lab", 14, "0.75, 0.45, 0.18", audio=aud, reference_clean_tail=ref_sub, shot_pipe=pipe_j, rife_bridge_mode="auto_jump_rife", bridge_max_frames=4)
        r_rep = pipe_j["last_seam_trim_report"]["rife"]
        self.assertTrue(r_rep["applied_rife"])
        self.assertEqual(r_rep["backend"], "rife")
        self.assertEqual(tuple(out_j.shape), tuple(curr_jump.shape))
        self.assertEqual(int(out_aud["waveform"].shape[-1]), int(aud["waveform"].shape[-1]))
        raw_jump_diff = float((curr_jump[0] - ref_sub[-1]).abs().mean().item())
        healed_jump_diff = float((out_j[0] - ref_sub[-1]).abs().mean().item())
        self.assertLess(healed_jump_diff, raw_jump_diff * 0.65)

    def test_single_video_o1_auto_heal_mode(self):
        trim_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSeamTrim"]()
        s1 = _make_textured_clip(32, shift_x=0.0, vx=0.6)
        s2 = _make_textured_clip(32, shift_x=32 * 0.6 + 12.0, vx=0.6, color_bias=(0.08, 0.0, -0.06))
        s3 = _make_textured_clip(32, shift_x=64 * 0.6 + 12.0, vx=0.6, color_bias=(0.12, -0.02, -0.09))
        s4 = _make_textured_clip(32, shift_x=96 * 0.6 + 24.0, vx=0.6, color_bias=(0.09, 0.01, -0.07))
        stitched = torch.cat([s1, s2, s3, s4], dim=0)
        aud = {"waveform": torch.zeros((1, 2, round(128 / 24.0 * 48000)), dtype=torch.float32), "sample_rate": 48000}

        out_img, out_aud = trim_n.trim_and_match(stitched, 0, 128, "single_video_auto_heal", 14, "0.75, 0.45, 0.18", audio=aud)
        self.assertEqual(tuple(out_img.shape), tuple(stitched.shape))
        self.assertEqual(int(out_aud["waveform"].shape[-1]), int(aud["waveform"].shape[-1]))
        self.assertLess(float((out_img[-1].mean(dim=(0, 1)) - s1[0].mean(dim=(0, 1))).abs().max().item()), 0.06)

    def test_flagship_v53_aesthetic_workflow_laws_bus_models_and_zero_aabb(self):
        p_ws = r"c:\Users\dcf\Documents\室内设计 琢影\ShotVault_v5.3_H3无劣化零色差RIFE无缝接续_旗舰美学版.json"
        p_wf = os.path.join(COMFY_ROOT, "user", "default", "workflows", "ShotVault_v5.3_H3无劣化零色差RIFE无缝接续_旗舰美学版.json")
        self.assertTrue(os.path.isfile(p_ws))
        self.assertTrue(os.path.isfile(p_wf))
        with open(p_ws, "rb") as f1, open(p_wf, "rb") as f2:
            self.assertEqual(hashlib.sha256(f1.read()).hexdigest(), hashlib.sha256(f2.read()).hexdigest())

        with open(p_wf, "r", encoding="utf-8") as f:
            wf = json.load(f)
        nodes = wf["nodes"]
        by_id = {n["id"]: n for n in nodes}
        groups_by_id = {g["id"]: g for g in wf["groups"]}

        # Top banner + Left/Right hemispheres + G1..G13 semantic colors
        self.assertIn(groups_by_id[99]["color"], ("#ffff85", "#747148"))
        self.assertGreaterEqual(groups_by_id[99].get("font_size", 24), 24)
        self.assertIn(100, groups_by_id)
        self.assertIn(200, groups_by_id)

        # Zero cross-hemisphere direct wires across Left (<1800) and Right (>=1900) hemispheres
        left_hi = groups_by_id[100]["bounding"][0] + groups_by_id[100]["bounding"][2]
        right_lo = groups_by_id[200]["bounding"][0]
        for lid, sid, _, did, _, _ in wf["links"]:
            sx, dx = by_id[sid]["pos"][0], by_id[did]["pos"][0]
            self.assertFalse((sx < left_hi and dx >= right_lo) or (sx >= right_lo and dx < left_hi), f"Cross link {lid}")
        self.assertEqual(groups_by_id[13]["bounding"][1], groups_by_id[12]["bounding"][1])
        self.assertGreater(groups_by_id[12]["bounding"][0], groups_by_id[13]["bounding"][0])
        self.assertGreater(groups_by_id[2]["bounding"][1], groups_by_id[13]["bounding"][1])
        for n in nodes:
            if n["pos"][0] < left_hi:
                self.assertNotIn(n["type"], {"easy forLoopStart", "easy forLoopEnd", "ShotVaultLoopDispatch", "ShotVaultLoad", "ShotVaultSave", "ShotVaultAssemble"})
            if n["type"] == "ShotVaultPlan":
                self.assertGreaterEqual(len(n["widgets_values"]), 18)
                self.assertEqual(n["widgets_values"][9], "")
                self.assertEqual(n["widgets_values"][10], "none")

        # Set/Get bus paired & collapsed
        sets = {n["widgets_values"][0] for n in nodes if n["type"] == "SetNode"}
        gets = {n["widgets_values"][0] for n in nodes if n["type"] == "GetNode"}
        self.assertTrue(gets.issubset(sets))
        self.assertTrue(all(n.get("flags", {}).get("collapsed") for n in nodes if n["type"] in ("SetNode", "GetNode")))

        # Upscalers, redundant canvas media loaders, T8 compat loaders & non-lazy switches completely removed
        self.assertNotIn(8, groups_by_id)
        for banned in (
            "ImageResizeKJv2",
            "VOSR2ModelLoader",
            "VOSR2Upscale",
            "Fast Groups Bypasser (rgthree)",
            "ComfySwitchNode",
            "LoadImage",
            "VHS_LoadVideo",
        ):
            self.assertFalse(any(n["type"] == banned for n in nodes), f"Unexpected/banned node {banned}")

        # Single 1101-style KJ preview monitor: vae/audio_vae unlinked, preview_frames synced dynamically, taeh3 + 24fps
        monitors = [n for n in nodes if n["type"] == "ModelPreviewOverrideKJ"]
        self.assertEqual(len(monitors), 1)
        mon = monitors[0]
        self.assertEqual(mon["widgets_values"], [1024, 80, True, 1, 24, "taeh3.safetensors", ""])
        mon_in = {inp["name"]: inp["link"] for inp in mon.get("inputs", [])}
        self.assertIsNone(mon_in.get("vae"))
        self.assertIsNone(mon_in.get("audio_vae"))
        self.assertIsNotNone(mon_in.get("preview_frames"))

        # 1101 3-screen SaveVideo monitoring (1 output slot), live concat, speed slider, regex & 4 SplitSigmas gears
        saves = [n for n in nodes if n["type"] == "SaveVideo"]
        self.assertEqual(len(saves), 3)
        for sv in saves:
            self.assertEqual(sv["widgets_values"][1:], ["auto", "auto", "auto"])
            self.assertEqual(len(sv.get("outputs", [])), 1)
        self.assertEqual(sum(1 for n in nodes if n["type"] == "ConcatenateVideo"), 1)
        self.assertEqual(sum(1 for n in nodes if n["type"] == "GoohaiUniversalSlider"), 1)
        self.assertEqual(sum(1 for n in nodes if n["type"] == "RegexExtract"), 1)
        self.assertEqual(sum(1 for n in nodes if n["type"] == "LoadAudio"), 1)
        splits = [n for n in nodes if n["type"] == "SplitSigmas"]
        self.assertEqual(len(splits), 4)
        self.assertEqual([s["widgets_values"][0] for s in splits], [8, 7, 7, 5])
        self.assertEqual(sum(1 for n in nodes if n["type"] == "ManualSigmas"), 4)

        trims = [n for n in nodes if n["type"] == "ShotVaultSeamTrim"]
        self.assertEqual(len(trims), 1)
        self.assertEqual(trims[0]["widgets_values"][2], "linear_22f_mkl_lab")
        self.assertEqual(trims[0]["widgets_values"][3], 22)
        self.assertEqual(trims[0]["widgets_values"][5], "off")
        bridges = [n for n in nodes if n["type"] == "AchengSeamBridge"]
        self.assertEqual(len(bridges), 1)
        bwv = bridges[0]["widgets_values"]
        self.assertEqual(bwv[0], "rife")
        self.assertEqual(bwv[2], 3)
        self.assertIs(bwv[5], False)
        self.assertIs(bwv[10], False)

        # Real local models & inputs
        models_root = os.path.join(COMFY_ROOT, "models")
        inp_root = os.path.join(COMFY_ROOT, "input")
        for n in nodes:
            t = n["type"]
            if t == "LoadAudio":
                self.assertTrue(os.path.isfile(os.path.join(inp_root, n["widgets_values"][0])))
            elif t == "UNETLoader":
                self.assertTrue(os.path.isfile(os.path.join(models_root, "unet", n["widgets_values"][0])))
            elif t == "CLIPLoader":
                self.assertTrue(os.path.isfile(os.path.join(models_root, "clip", n["widgets_values"][0])))
            elif t == "VAELoader":
                self.assertTrue(os.path.isfile(os.path.join(models_root, "vae", n["widgets_values"][0])))
            elif t == "LoraLoaderModelOnly":
                self.assertTrue(os.path.isfile(os.path.join(models_root, "loras", n["widgets_values"][0])))
            elif t == "MiniMaxH3SemanticBridgeConfigT8":
                self.assertTrue(os.path.isfile(os.path.join(models_root, "semantic_bridge", n["widgets_values"][0])))

        # Zero AABB overlap across all nodes
        for i in range(len(nodes)):
            a = nodes[i]
            ax1, ay1 = a["pos"]
            ax2, ay2 = ax1 + a["size"][0], ay1 + a["size"][1]
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                bx1, by1 = b["pos"]
                bx2, by2 = bx1 + b["size"][0], by1 + b["size"][1]
                self.assertTrue(ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1, f"Overlap #{a['id']} vs #{b['id']}")

    def test_global_22_vs_39_gear_governs_3shot_external_loop_end_to_end(self):
        import comfy.nested_tensor  # type: ignore

        taper_cls = importlib.import_module("ComfyUI-H3-Context-Noise.nodes").MiniMaxH3ContextTaperNoise
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        disp_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoopDispatch"]()
        wg_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        trim_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSeamTrim"]()
        save_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()
        asm_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultAssemble"]()

        class _MockAudioVAE:
            audio_sample_rate = 32000

            def encode(self, wav):
                return torch.ones((1, 32, 2, max(1, round(int(wav.shape[1]) / 32000.0 * 40.0))), dtype=torch.float32) * 0.25

        class _MockVideoVAE:
            def __init__(self):
                self.calls = 0

            def encode(self, frames):
                self.calls += 1
                f = int(frames.shape[0])
                t = 2 if f <= 5 else ((f - 5) // 17) * 5 + 2
                return torch.ones((1, 24, t, max(1, int(frames.shape[1]) // 16), max(1, int(frames.shape[2]) // 16)), dtype=torch.float32) * 0.5

        taper = taper_cls()
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        script = "& 3.0 &\n一段\n===\n& 3.0 &\n二段\n===\n& 3.0 &\n三段"

        for gear_str, exp_tok1 in (("22", 7), ("39", 12)):
            gear = int(gear_str)
            with tempfile.TemporaryDirectory() as tmp, patch("folder_paths.get_output_directory", return_value=tmp):
                pipe0, loop_total, _ = plan_cls.plan(f"LoopGear{gear}", "auto", "film", script, "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", gear_str)
                self.assertEqual(loop_total, 3)
                carry = pipe0
                for idx in range(loop_total):
                    res = disp_cls.dispatch(pipe0, segment_index=idx, index_base="0_based (easy_forLoop)", loop_carry=carry)
                    sp, prev_clean = res[0], res[26]
                    if idx > 0:
                        self.assertEqual(int(res[10]), gear)
                        self.assertEqual(int(res[11]), gear)
                        self.assertEqual(int(res[12]), gear)
                        self.assertEqual(int(prev_clean.shape[0]), gear)
                    tapered, _ = taper.inject(prev_clean, res[12], 0.55, 0.0, res[13], 0)
                    raw_frames = int(sp["target_frames"])
                    vt = 2 if raw_frames <= 5 else ((raw_frames - 5) // 17) * 5 + 2
                    lat = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, vt, 2, 2)), torch.zeros((1, 32, 2, 93))))}
                    vvae = _MockVideoVAE()
                    avae = _MockAudioVAE()
                    g1, _ = wg_cls.prepare_guider(None, pos, lat, "single_or_1st", "off", 192, gear, "auto", 0.0, 7, True, False, 1.0, vae=vvae, audio_vae=avae, tapered_prev_images=tapered, shot_pipe=sp)
                    g2, _ = wg_cls.prepare_guider(None, pos, lat, "2nd_refine", "off", 192, gear, "auto", 0.0, 7, True, False, 1.0, vae=vvae, audio_vae=avae, tapered_prev_images=tapered, shot_pipe=sp)
                    if idx > 0:
                        self.assertEqual(vvae.calls, 1)  # Encoded once in Pass 1; Pass 2 reuses cached full head Latent!
                        kfs1 = [k for k in g1[0][1]["minimax_keyframes"] if k.get("latent") is not None]
                        kfs2 = [k for k in g2[0][1]["minimax_keyframes"] if k.get("latent") is not None]
                        self.assertEqual(kfs1[0]["resolved_frame_index"], 0)
                        self.assertEqual(kfs2[0]["resolved_frame_index"], 0)
                        vk1, vk2 = kfs1[0]["latent"], kfs2[0]["latent"]
                        self.assertEqual(vk1.shape[2], exp_tok1)
                        self.assertEqual(vk2.shape[2], exp_tok1)
                        self.assertTrue(torch.equal(vk1, vk2))
                    self.assertEqual(int(sp["trim_head_frames"]), gear if idx > 0 else 0)
                    fake_dec = _make_textured_clip(raw_frames, h=64, w=64)
                    trimmed, out_aud = trim_cls.trim_and_match(fake_dec, -1, -1, "auto_dual_mkl_lab", 14, "0.75, 0.45, 0.18", reference_clean_tail=prev_clean, shot_pipe=sp)
                    self.assertEqual(int(trimmed.shape[0]), int(sp["keep_frames"]))
                    _, _, carry, _, _, _ = save_cls.save_take(trimmed, sp, False, 20, "h264", final_audio=out_aud, clean_images=trimmed)["result"]
                m_path, _, _, _ = asm_cls.assemble(f"LoopGear{gear}", "defer_until_complete", "auto_prefer_reference", 15.0, 0.0, True, True, shot_pipe=carry)["result"]
                self.assertTrue(os.path.isfile(m_path))

        with tempfile.TemporaryDirectory() as tmp, patch("folder_paths.get_output_directory", return_value=tmp):
            dplan = json.dumps({
                "context_frames": 39,
                "retry_lock": {"enabled": True, "shot_index": 2, "seed_mode": "randomize"},
                "shots": [
                    {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "hard_cut": True, "prompt": "A"},
                    {"shot_index": 2, "duration_sec": 3.0, "shot_type": 0, "hard_cut": False, "prompt": "B"},
                ],
            })
            pipe_rl, rl_total, _ = plan_cls.plan("Retry1", "auto", "film", "", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "39", director_plan_json=dplan)
            self.assertEqual(rl_total, 1)
            self.assertEqual(len(pipe_rl["shots"]), 2)

    def test_v55_regular_mode_override_delta_e_after_ref_order_and_v3_workflow_schemas(self):
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        cond_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultH3Conditioning"]()

        # 1. Explicit audio_drive_mode="none" in director_plan_json overrides leftover per-shot guide_audio
        with tempfile.TemporaryDirectory() as tmp, patch("folder_paths.get_output_directory", return_value=tmp):
            dplan = json.dumps({
                "audio_drive_mode": "none",
                "context_frames": 22,
                "shots": [
                    {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "prompt": "动作", "guide_audio": "shotvault/stale_guide.wav"},
                ],
            })
            pipe, _, _ = plan_cls.plan(
                "RegOverride", "auto", "c", "", "16:9 (1344x768)", 0.25, 24,
                "cumulative_cursor", "22", audio_drive_mode="none", director_plan_json=dplan,
            )
            self.assertFalse(pipe["audio_drive_active"])
            self.assertEqual(pipe["audio_drive_mode"], "none")
            self.assertEqual(pipe["context_frames"], 22)

        # 2. auto_dual_mkl_lab_match reports delta_e_after <= delta_e_eff
        ref = _make_textured_clip(16, h=48, w=48, color_bias=(0.0, 0.0, 0.0))
        drifted = _make_textured_clip(16, h=48, w=48, color_bias=(0.14, -0.05, -0.10))
        _, rep = sc_mod.auto_dual_mkl_lab_match(drifted, ref)
        self.assertTrue(rep["applied_color"])
        self.assertIn("delta_e_after", rep)
        self.assertLess(rep["delta_e_after"], rep["delta_e_eff"])

        # 3. Director UI ref images precede Group 4 extra_ref_image_1 so <Picture 1> stays intact
        class _CapClip:
            def tokenize(self, prompt, **kwargs):
                return [[prompt, kwargs]]
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros((1, 2, 4)), {"tok": tokens}]]

        ui_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        ext_img = torch.ones((1, 64, 64, 3), dtype=torch.float32)
        pipe_refs = {
            "prompt": "<Picture 1> hero",
            "width": 128,
            "height": 128,
            "target_frames": 39,
            "loaded_ref_images": [ui_img],
            "loaded_ref_videos": [],
            "loaded_ref_audios": [],
        }
        _, _, ref_map = cond_cls.encode_shot(_CapClip(), pipe_refs, "match", extra_ref_image_1=ext_img)
        self.assertIn("<Picture 1>=shot_img_1", ref_map)
        self.assertIn("<Picture 2>=canvas_img_1", ref_map)

        # 4. Flagship workflow schema alignment & pre-populated 2-shot director_plan_json
        p_wf = os.path.join(COMFY_ROOT, "user", "default", "workflows", "ShotVault_v5.3_H3无劣化零色差RIFE无缝接续_旗舰美学版.json")
        with open(p_wf, "r", encoding="utf-8") as f:
            wf = json.load(f)
        by_type = {}
        for n in wf["nodes"]:
            by_type.setdefault(n["type"], []).append(n)
        self.assertEqual(by_type["CreateVideo"][0]["widgets_values"], [24, 8, "sRGB", "auto", "h264"])
        self.assertEqual(len(by_type["SaveVideo"]), 3)
        for sv in by_type["SaveVideo"]:
            self.assertEqual(sv["widgets_values"][1:], ["auto", "auto", "auto"])
        self.assertNotIn("ImageResizeKJv2", by_type)
        self.assertNotIn("VOSR2Upscale", by_type)
        self.assertEqual(by_type["ShotVaultPlan"][0]["widgets_values"][7], "zero_tail_trim")
        self.assertEqual(by_type["ShotVaultPlan"][0]["widgets_values"][8], "22")
        for wg in by_type["ShotVaultWindowGuider"]:
            self.assertEqual(wg["widgets_values"][1], "auto_over_260f")
            self.assertEqual(wg["widgets_values"][2], 260)
            self.assertEqual(wg["widgets_values"][4], "auto")
            self.assertEqual(wg["widgets_values"][5], 0.0)
            self.assertEqual(wg["widgets_values"][6], 7)
        self.assertEqual(len(by_type["SplitSigmas"]), 4)
        self.assertNotIn("ComfySwitchNode", by_type)
        self.assertIn(by_type["ShotVaultPlan"][0]["widgets_values"][0], ("Flagship_v56_Test", "Flagship_v56_SeamBridge"))
        self.assertIn(round(float(by_type["ShotVaultPlan"][0]["widgets_values"][5]), 2), (0.25, 0.40, 1.00))
        plan_wf = json.loads(by_type["ShotVaultPlan"][0]["widgets_values"][17])
        self.assertEqual(len(plan_wf.get("shots", [])), 23)
        self.assertEqual(len([x for x in plan_wf.get("global_ref_slots", []) if x]), 9)
        inp_root = os.path.join(COMFY_ROOT, "input")
        for sh in plan_wf["shots"]:
            self.assertEqual(sh["shot_type"], 1)
            self.assertEqual(len([x for x in sh.get("ref_slots", []) if x]), 9)
            for rel in sh["ref_slots"]:
                self.assertTrue(os.path.isfile(os.path.join(inp_root, rel)), rel)

        # 5. preview_target_frames exported by ShotVaultPlan (both max & single-shot retry target) & .txt sidecar archival in ShotVaultSave
        with tempfile.TemporaryDirectory() as tmp, patch("folder_paths.get_output_directory", return_value=tmp):
            dplan_multi = json.dumps({
                "audio_drive_mode": "none",
                "context_frames": 22,
                "shots": [
                    {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "prompt": "动作A"},
                    {"shot_index": 2, "duration_sec": 6.0, "shot_type": 0, "prompt": "文戏B"},
                ],
            })
            _, total_m, sum_m_str = plan_cls.plan(
                "PreviewSync", "auto", "c", "", "16:9 (1344x768)", 0.25, 24,
                "cumulative_cursor", "22", audio_drive_mode="none", director_plan_json=dplan_multi,
            )
            sum_m = json.loads(sum_m_str)
            self.assertEqual(total_m, 2)
            self.assertEqual(len(sum_m["target_frames_per_shot"]), 2)
            self.assertEqual(sum_m["preview_target_frames"], max(sum_m["target_frames_per_shot"]))

            dplan_retry = json.dumps({
                "audio_drive_mode": "none",
                "context_frames": 22,
                "retry_lock": {"enabled": True, "shot_index": 1},
                "shots": [
                    {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "prompt": "动作A"},
                    {"shot_index": 2, "duration_sec": 6.0, "shot_type": 0, "prompt": "文戏B"},
                ],
            })
            _, total_r, sum_r_str = plan_cls.plan(
                "PreviewSync", "auto", "c", "", "16:9 (1344x768)", 0.25, 24,
                "cumulative_cursor", "22", audio_drive_mode="none", director_plan_json=dplan_retry,
            )
            sum_r = json.loads(sum_r_str)
            self.assertEqual(total_r, 1)
            self.assertEqual(sum_r["preview_target_frames"], sum_r["target_frames_per_shot"][0])

            save_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()
            imgs = _make_textured_clip(12, h=48, w=64)
            s_pipe = {
                "project_name": "PreviewSync",
                "project_root": sum_m["project_root"],
                "fps": 24,
                "current_shot_index": 1,
                "seed_text": "123456",
                "overlap_frames": 22,
                "trim_head_frames": 4,
                "keep_frames": 12,
                "shot_type": 1,
                "prompt": "测试逐段配方归档",
                "active_ref_summary": "<Picture 1>=hero.png",
                "last_seam_trim_report": {"applied_color": False, "delta_e_eff": 0.4},
                "loaded_ref_images": [torch.ones((1, 16, 16, 3))],
                "delivery_ref_audio": {"waveform": torch.ones((1, 2, 4800)), "sample_rate": 48000},
            }
            save_res = save_cls.save_take(imgs, s_pipe, False, 18, "h264")
            _, _, updated_pipe, take_id, vid_path = save_res["result"]
            self.assertTrue(os.path.isfile(vid_path))
            self.assertNotIn("loaded_ref_images", updated_pipe)
            self.assertNotIn("delivery_ref_audio", updated_pipe)
            txt_path = os.path.join(sum_m["project_root"], "shots", f"{take_id}.txt")
            self.assertTrue(os.path.isfile(txt_path))
            with open(txt_path, "r", encoding="utf-8") as sf:
                txt = sf.read()
            self.assertIn("seed=123456", txt)
            self.assertIn("gear_overlap=22f | trim_head=4f | keep_frames=12f", txt)
            self.assertIn("seam_trim_v53=", txt)
            self.assertIn("测试逐段配方归档", txt)

    def test_audio_mv_conflict_guard_and_48k_monitor_audio(self):
        import wave

        asset_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.asset_and_turnaround")
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        cond_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultH3Conditioning"]()
        trim_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSeamTrim"]()

        # 1. master_song_slice without master_audio_in or per-shot audio raises clear Chinese ValueError
        with tempfile.TemporaryDirectory() as tmp, patch("folder_paths.get_output_directory", return_value=tmp):
            with self.assertRaises(ValueError) as ctx:
                plan_cls.plan(
                    "MissingMaster", "auto", "c", "& 3.0 &\nA", "16:9 (1344x768)", 0.25, 24,
                    "cumulative_cursor", "39", audio_drive_mode="master_song_slice", master_audio_in=None,
                )
            self.assertIn("整曲母带音频", str(ctx.exception))

            # 2. Regular mode ("none") safely ignores attached master_audio_in
            song = torch.sin(torch.linspace(0.0, 20.0, 48000 * 4)).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
            pipe_reg, _, _ = plan_cls.plan(
                "RegIgnoresMaster", "auto", "c", "& 3.0 &\nA", "16:9 (1344x768)", 0.25, 24,
                "cumulative_cursor", "22", audio_drive_mode="none",
                master_audio_in={"waveform": song, "sample_rate": 48000},
            )
            self.assertFalse(pipe_reg["audio_drive_active"])
            self.assertEqual(pipe_reg["per_shot_audio"], {})

            # 3. skip_ref_audios=True omits <Audio j> even when wav file exists on disk
            wav_p = os.path.join(tmp, "voice_ref.wav")
            with wave.open(wav_p, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x01\x00" * 1600)
            mm_reg = asset_mod.load_shot_multimodal_refs(
                {"prompt": "hi <Audio 1>", "ref_audios": [wav_p]}, input_dir=tmp, project_root=tmp, skip_ref_audios=False,
            )
            mm_mv = asset_mod.load_shot_multimodal_refs(
                {"prompt": "hi <Audio 1>", "ref_audios": [wav_p]}, input_dir=tmp, project_root=tmp, skip_ref_audios=True,
            )
            self.assertEqual(len(mm_reg["audios"]), 1)
            self.assertEqual(len(mm_mv["audios"]), 0)
            self.assertNotIn("<Audio", mm_mv["summary"])

        # 4. ShotVaultH3Conditioning strips <Audio N> tags & blocks ref audios/video soundtracks when audio_drive_active=True
        captured: list[tuple[str, object]] = []

        class _CapClip:
            def tokenize(self, prompt, **kwargs):
                captured.append((prompt, kwargs.get("minimax_ref_items")))
                return [[prompt, kwargs]]

            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros((1, 2, 4)), {"tok": tokens}]]

        aud_ref = {"waveform": torch.ones((1, 2, 1600), dtype=torch.float32) * 0.2, "sample_rate": 16000}
        vid_ref = {"frames": torch.zeros((5, 64, 64, 3), dtype=torch.float32), "audio": aud_ref, "name": "rv.mp4"}
        pipe_mv = {
            "prompt": "<Picture 1> warrior <Audio 1> punches <Video 1>",
            "audio_drive_active": True,
            "width": 128,
            "height": 128,
            "target_frames": 39,
            "loaded_ref_images": [torch.zeros((1, 64, 64, 3))],
            "loaded_ref_videos": [vid_ref],
            "loaded_ref_audios": [aud_ref],
        }
        cond, _, ref_map = cond_cls.encode_shot(_CapClip(), pipe_mv, "match", extra_ref_audio_1=aud_ref)
        self.assertEqual(len(captured), 1)
        self.assertNotIn("<Audio", captured[0][0])
        items = captured[0][1] or []
        self.assertFalse(any(it.get("type") == "audio" for it in items))
        self.assertNotIn("<Audio", ref_map)

        # 5. ShotVaultSeamTrim outputs 48kHz delivery_ref_audio in audio_drive_active mode even when decoded audio is present
        imgs_t = _make_textured_clip(12, h=48, w=48)
        decoded_aud = {"waveform": torch.zeros((1, 2, 16000), dtype=torch.float32), "sample_rate": 32000}
        master_slice = {"waveform": torch.ones((1, 2, 24000), dtype=torch.float32) * 0.35, "sample_rate": 48000}
        _, out_aud = trim_cls.trim_and_match(
            imgs_t, 0, 12, "off", 14, "0.75, 0.45, 0.18",
            audio=decoded_aud,
            shot_pipe={"fps": 24, "is_continuation": False, "audio_drive_active": True, "delivery_ref_audio": master_slice},
            rife_bridge_mode="off",
        )
        self.assertEqual(int(out_aud["sample_rate"]), 48000)
        self.assertAlmostEqual(float(out_aud["waveform"].mean().item()), 0.35, places=3)

    def test_v55_symmetric_auto_head_latent_and_zero_sigma_drop_across_two_pass(self):
        import comfy.nested_tensor  # type: ignore
        cw_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.context_window_guider")
        wg_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()

        class _NeverEncodeVAE:
            def encode(self, _frames):
                raise AssertionError("vae.encode must NOT be called when bare prev_tail_latent matches!")

        class _StubModel:
            def __init__(self):
                self.wrappers = {}
            def clone(self):
                return _StubModel()

        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        lat = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 17, 2, 2)), torch.zeros((1, 32, 2, 93))))}
        bare_tail = {
            "samples": comfy.nested_tensor.NestedTensor((
                torch.arange(24 * 12 * 2 * 2, dtype=torch.float32).reshape(1, 24, 12, 2, 2),
                torch.zeros((1, 32, 2, 65), dtype=torch.float32),
            )),
            "shotvault_meta": {"audio_overhang": 0.0},
        }
        sp = {"shot_index": 2, "parent_take_id": "s1_t1", "is_continuation": True, "overlap_frames": 39, "context_frames": 39}
        g1, _ = wg_cls.prepare_guider(None, pos, lat, "single_or_1st", "off", 192, 39, "auto", 0.0, 7, True, False, 1.0, vae=_NeverEncodeVAE(), prev_tail_latent=bare_tail, shot_pipe=sp)
        g2, _ = wg_cls.prepare_guider(None, pos, lat, "2nd_refine", "off", 192, 39, "auto", 0.0, 7, True, False, 1.0, vae=_NeverEncodeVAE(), prev_tail_latent=bare_tail, shot_pipe=sp)
        k1 = [k for k in g1[0][1]["minimax_keyframes"] if k.get("latent") is not None][0]
        k2 = [k for k in g2[0][1]["minimax_keyframes"] if k.get("latent") is not None][0]
        self.assertEqual(k1["resolved_frame_index"], 0)
        self.assertEqual(k2["resolved_frame_index"], 0)
        self.assertEqual(k1["latent"].shape[2], 12)  # 39f = 34+5 -> 12 latent tokens
        self.assertTrue(torch.equal(k1["latent"], k2["latent"]))
        # keyframe_until_sigma=0.0 must be a complete no-op so Pass 2 (<0.85) never drops continuation
        m = _StubModel()
        self.assertIs(cw_mod.apply_keyframe_until(m, 0.0), m)
        self.assertNotIn("shotvault_keyframe_until", m.wrappers)

    def test_v56_zero_tail_trim_260f_single_window_and_known_continuation_seam_rescue(self):
        import comfy.nested_tensor  # type: ignore
        import comfy.samplers  # type: ignore
        gm = importlib.import_module("ComfyUI-Acheng-ShotVault.core.grid_math")
        wg_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        trim_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSeamTrim"]()

        # 1. zero_tail_trim on [8.0, 8.0] matches 1101 exact 204f retention and NEVER triggers tail_trimmed
        s22 = gm.compute_cumulative_schedule([8.0, 8.0], fps=24, context_frames=22, duration_mode="zero_tail_trim")
        self.assertEqual((s22[0]["target_frames"], s22[0]["keep_frames"], s22[0]["trim_head_frames"]), (192, 192, 0))
        self.assertEqual((s22[1]["target_frames"], s22[1]["keep_frames"], s22[1]["trim_head_frames"]), (226, 204, 22))
        s39 = gm.compute_cumulative_schedule([8.0, 8.0], fps=24, context_frames=39, duration_mode="zero_tail_trim")
        self.assertEqual((s39[1]["target_frames"], s39[1]["keep_frames"], s39[1]["trim_head_frames"]), (243, 204, 39))
        self.assertEqual(gm.resolve_duration_mode("zero_tail_trim", audio_drive_active=True), "cumulative_cursor")
        with tempfile.TemporaryDirectory() as tmp:
            store = vs_mod.VaultStore(tmp)
            lat_243 = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 72, 2, 2)), torch.zeros((1, 32, 2, 405))))}
            meta2 = store.save_take_context(
                take_id="s2_t1", shot_index=2, clean_images=_make_textured_clip(204, h=64, w=64),
                audio_waveform_48k=torch.zeros((2, 408000)), audio_sr=48000, fps=24, prompt_sha256="x",
                parent_take_id="s1_t1", seed_text="42", keep_frames=204, trim_head_frames=39, shot_type=1,
                sampled_latent=lat_243, tail_latent_frames=39, raw_trimmed_frames=204,
            )
            self.assertTrue(meta2["has_tail_latent"])
            self.assertEqual(meta2["tail_latent_skipped_reason"], "")

        # 2. auto_over_260f avoids splitting an 8s+39f (243f) segment into sub-windows, while 396f still uses ContextWindows
        import comfy.model_base  # type: ignore
        import comfy.model_patcher  # type: ignore
        import comfy.supported_models  # type: ignore
        cfg_m = comfy.supported_models.MiniMaxH3({"image_model": "minimax_h3", "disable_unet_model_creation": True})
        base_m = comfy.model_base.MiniMaxH3(cfg_m, device=torch.device("cpu"))
        patcher_m = comfy.model_patcher.ModelPatcher(base_m, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        lat_243_s = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 72, 2, 2)), torch.zeros((1, 32, 2, 405))))}
        lat_396_s = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 117, 2, 2)), torch.zeros((1, 32, 2, 660))))}
        g_243, _ = wg_cls.prepare_guider(patcher_m, pos, lat_243_s, "single_or_1st", "auto_over_260f", 260, 39, "auto", 0.0, 7, True, False, 1.0)
        g_396, _ = wg_cls.prepare_guider(patcher_m, pos, lat_396_s, "single_or_1st", "auto_over_260f", 260, 39, "auto", 0.0, 7, True, False, 1.0)
        self.assertIs(type(g_243), comfy.samplers.CFGGuider)
        self.assertFalse(hasattr(g_243, "window_set"))
        self.assertTrue(hasattr(g_396, "window_set") and len(g_396.window_set.windows) >= 2)

        # 3. Known continuation rescues high-motion martial-arts seams (low patch cov, high luma_err, dE>5) instead of false hard_cut_bypass
        ref_act = _make_textured_clip(10, h=64, w=64, shift_x=0.0, vx=3.0)
        curr_act = _make_textured_clip(16, h=64, w=64, shift_x=38.0, vx=3.0, color_bias=(0.14, -0.05, -0.10))
        curr_act[0] = torch.roll(curr_act[0], shifts=(0, 18), dims=(0, 1))
        pipe_cont = {"fps": 24, "trim_head_frames": 0, "keep_frames": 16, "is_continuation": True}
        _, _ = trim_n.trim_and_match(curr_act, -1, -1, "auto_dual_mkl_lab", 14, "0.75, 0.45, 0.18", reference_clean_tail=ref_act, shot_pipe=pipe_cont, rife_bridge_mode="auto_jump_rife")
        creps = pipe_cont["last_seam_trim_report"]["color"]
        rreps = pipe_cont["last_seam_trim_report"]["rife"]
        self.assertTrue(creps["same_scene_safe"])
        self.assertTrue(creps["applied_color"])
        self.assertLess(creps["delta_e_after"], creps["delta_e_eff"] * 0.5)
        self.assertNotEqual(rreps["reason"], "hard_cut_protected")
        # 4. Wide camera-angle jump (same-palette spatial rotation, d_seam > 0.16) still locks color cleanly while blocking RIFE smearing
        wide_cut = (0.42 + (torch.rot90(ref_act[-1:], 2, dims=(1, 2)) - 0.42) * 2.6).clamp(0.02, 0.98).repeat(12, 1, 1, 1)
        pipe_wc = {"fps": 24, "trim_head_frames": 0, "keep_frames": 12, "is_continuation": True}
        _, _ = trim_n.trim_and_match(wide_cut, -1, -1, "auto_dual_mkl_lab", 14, "0.75, 0.45, 0.18", reference_clean_tail=ref_act, shot_pipe=pipe_wc, rife_bridge_mode="auto_jump_rife")
        self.assertTrue(pipe_wc["last_seam_trim_report"]["color"]["same_scene_safe"])
        self.assertFalse(pipe_wc["last_seam_trim_report"]["rife"]["applied_rife"])
        self.assertEqual(pipe_wc["last_seam_trim_report"]["rife"]["reason"], "camera_cut_too_wide_for_rife")


if __name__ == "__main__":
    unittest.main()
