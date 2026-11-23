"""Phase 11 tests: Storyboard Planner renaming, in-panel async/atomic master export,
@cut@/@continue@ dual-syntax parsing + continuation depth, and fixed-camera digital-human anti-degradation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import av
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

SPEC = importlib.util.spec_from_file_location(
    "shotvault_pkg_p11", os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT]
)
assert SPEC and SPEC.loader
SV = importlib.util.module_from_spec(SPEC)
sys.modules["shotvault_pkg_p11"] = SV
SPEC.loader.exec_module(SV)

from shotvault_pkg_p11.core.context_window_guider import _apply_anti_degradation_guide
from shotvault_pkg_p11.core.script_parser import parse_director_plan_json, parse_script
from shotvault_pkg_p11.core.seam_qc import write_mp4_with_pcm
from shotvault_pkg_p11.core.vault_store import VaultStore, compute_continuation_depth
from shotvault_pkg_p11.server_routes import _build_project_state_payload, handle_assemble_request


class TestPhase11PlannerExportCutAndAntiDeg(unittest.TestCase):
    def test_planner_renaming_no_director_console_in_ui(self):
        disp = SV.NODE_DISPLAY_NAME_MAPPINGS["ShotVaultPlan"]
        self.assertIn("分镜规划", disp)
        self.assertNotIn("导演台", disp)

        i18n_js = open(os.path.join(ROOT, "web", "i18n.js"), "r", encoding="utf-8").read()
        self.assertIn("ShotVault 分镜规划", i18n_js)
        self.assertIn("ShotVault Storyboard Planner", i18n_js)
        self.assertNotIn("导演台", i18n_js)

        panel_js = (
            open(os.path.join(ROOT, "web", "director_panel.js"), "r", encoding="utf-8").read()
            + "\n"
            + open(os.path.join(ROOT, "web", "panel", "export_modal.js"), "r", encoding="utf-8").read()
        )
        self.assertIn("renderExportModal", panel_js)
        self.assertIn("/shotvault/assemble", panel_js)
        self.assertIn("computeContinuationStreaks", panel_js)
        self.assertIn("digital_human_static_bg", panel_js)

    def test_cut_continue_trailing_at_and_continuation_depth(self):
        script = (
            "&10&\n@cut@\n首镜开场画面 contact test@cut.com\n"
            "===\n&8&\n@continue@\n第二段续接画面\n"
            "===\n&6&\n@continue\n第三段续接画面\n"
            "===\n&6&\n@cut@\n第四段切机位清零\n"
            "===\n&6&\n@continue@ 第五段内联续接\n"
        )
        shots = parse_script(script, common_prompt="", fps=24, has_seed_video=False)
        self.assertEqual(len(shots), 5)
        self.assertEqual([s["is_continuation"] for s in shots], [False, True, True, False, True])
        # Zero stray '@' left from @cut@ / @continue@ while preserving normal email address
        self.assertIn("test@cut.com", shots[0]["local_prompt"])
        for i in range(1, 5):
            self.assertNotIn("@", shots[i]["local_prompt"])
        self.assertTrue(shots[4]["local_prompt"].startswith("第五段内联续接"))

        depths = [compute_continuation_depth(shots, s["shot_index"], s["is_continuation"]) for s in shots]
        self.assertEqual(depths, [1, 2, 3, 1, 2])

        # Verify anti_degradation config parsing & bounds clamping
        gcfg, _ = parse_director_plan_json(
            json.dumps({
                "anti_degradation": {"mode": "digital_human_static_bg", "noise_keep_ratio": 0.35, "static_bg_pull": 0.99},
                "shots": [{"duration_sec": 5.0, "prompt": "hi"}],
            }),
            common_prompt="",
            fps=24,
        )
        self.assertEqual(gcfg["anti_degradation"]["mode"], "digital_human_static_bg")
        self.assertAlmostEqual(gcfg["anti_degradation"]["noise_keep_ratio"], 0.35, places=4)
        self.assertAlmostEqual(gcfg["anti_degradation"]["static_bg_pull"], 0.35, places=4)  # hard-clamped <= 0.35

    def test_anti_degradation_standard_low_noise_and_static_bg_gates(self):
        torch.manual_seed(7)
        f_cnt, h, w = 16, 128, 192
        # Clean Shot 1 anchor background = uniform 0.30 gray-blue
        anchor = torch.full((4, h, w, 3), 0.30, dtype=torch.float32)

        # Clean tail at Shot 5: background drifted slightly (+0.05 -> 0.35), central face patch oscillates strongly (moving!)
        clean_tail = torch.full((f_cnt, h, w, 3), 0.35, dtype=torch.float32)
        for t_i in range(f_cnt):
            phase = 0.25 if (t_i % 2 == 0) else 0.75
            clean_tail[t_i, 44:84, 76:116, :] = phase  # active moving foreground (mouth/head)

        # Simulated #106 TaperNoise adding +/-0.15 chroma blocks everywhere
        noise = torch.empty_like(clean_tail).uniform_(-0.15, 0.15)
        tapered = (clean_tail + noise).clamp(0.0, 1.0)

        # 1. Standard mode -> bit-for-bit identical passthrough
        out_std, info_std = _apply_anti_degradation_guide(tapered, clean_tail, anchor, {"mode": "standard"}, continuation_depth=5)
        self.assertTrue(torch.equal(out_std, tapered))
        self.assertEqual(info_std["applied"], "passthrough")

        # 2. Digital human low-noise -> noise amplitude linearly reduced to lambda=0.40
        out_low, info_low = _apply_anti_degradation_guide(
            tapered, clean_tail, anchor, {"mode": "digital_human_low_noise", "noise_keep_ratio": 0.40}, continuation_depth=5
        )
        self.assertEqual(info_low["applied"], "low_noise_only")
        orig_err = (tapered - clean_tail).abs().mean().item()
        low_err = (out_low - clean_tail).abs().mean().item()
        self.assertLess(low_err, orig_err * 0.45)

        # 3. Digital human static-bg on fixed camera -> pulls static corners toward anchor (0.30), protects moving center!
        out_bg, info_bg = _apply_anti_degradation_guide(
            tapered, clean_tail, anchor, {"mode": "digital_human_static_bg", "noise_keep_ratio": 0.36, "static_bg_pull": 0.25}, continuation_depth=5
        )
        self.assertEqual(info_bg["applied"], "low_noise_and_static_bg_anchor")
        out_low36, _ = _apply_anti_degradation_guide(
            tapered, clean_tail, anchor, {"mode": "digital_human_low_noise", "noise_keep_ratio": 0.36}, continuation_depth=5
        )
        # Corners (static bg) must move measurably closer to 0.30 anchor
        corner_before = (out_low36[:, :20, :20, :] - 0.30).abs().mean().item()
        corner_after = (out_bg[:, :20, :20, :] - 0.30).abs().mean().item()
        self.assertLess(corner_after, corner_before - 0.008)
        # Moving foreground center (54:74, 86:106) must be 100% protected from anchor ghosting!
        center_diff = (out_bg[:, 54:74, 86:106, :] - out_low36[:, 54:74, 86:106, :]).abs().max().item()
        self.assertLess(center_diff, 1e-4)

        # 4. Camera moved / cut -> global correlation gate auto-bypasses static-bg pull (zero ghosting!)
        moved_anchor = torch.full((4, h, w, 3), 0.78, dtype=torch.float32)
        out_mv, info_mv = _apply_anti_degradation_guide(
            tapered, clean_tail, moved_anchor, {"mode": "digital_human_static_bg", "noise_keep_ratio": 0.36, "static_bg_pull": 0.25}, continuation_depth=5
        )
        self.assertEqual(info_mv.get("static_bg_skipped"), "camera_moved")
        self.assertTrue(torch.allclose(out_mv, out_low36, atol=1e-6))

    def test_in_panel_assemble_handles_stale_seam_repairs_and_timestamped_export(self):
        with tempfile.TemporaryDirectory() as td:
            # Point resolve_output_root via monkeypatching server_routes.resolve_output_root to td
            import shotvault_pkg_p11.server_routes as sr_mod

            orig_resolve = sr_mod.resolve_output_root
            sr_mod.resolve_output_root = lambda name: os.path.join(td, name)
            try:
                proj_name = "TestExportProj"
                proj_root = os.path.join(td, proj_name)
                store = VaultStore(proj_root)
                rng = np.random.default_rng(42)
                base = torch.from_numpy(rng.uniform(0.25, 0.75, (1, 128, 192, 3)).astype(np.float32))
                img1 = base.repeat(36, 1, 1, 1)
                img2 = torch.clamp(base.repeat(36, 1, 1, 1) + 0.02, 0.0, 1.0)
                img3 = torch.clamp(base.repeat(36, 1, 1, 1) + 0.04, 0.0, 1.0)
                n_s = int(round(36 / 24.0 * 48000))
                t_v = torch.linspace(0.0, 1.5, n_s, dtype=torch.float32)
                tone = 0.08 * torch.sin(2.0 * np.pi * 330.0 * t_v)
                aud = torch.stack([tone, tone], dim=0)

                for idx, (tid, img, parent) in enumerate([
                    ("shot_0001_take_001", img1, ""),
                    ("shot_0002_take_002", img2, "shot_0001_take_001"),
                    ("shot_0003_take_001", img3, "shot_0002_take_001"),  # parent mismatch -> STALE!
                ], start=1):
                    vrel = f"shots/{tid}.mp4"
                    write_mp4_with_pcm(img, aud, 48000, 24, os.path.join(proj_root, vrel))
                    store.save_take_context(
                        take_id=tid, shot_index=idx, clean_images=img, audio_waveform_48k=aud,
                        audio_sr=48000, fps=24, prompt_sha256=f"p{idx}", parent_take_id=parent,
                        seed_text=str(idx), keep_frames=36, trim_head_frames=0, shot_type=1,
                        video_rel_path=vrel, set_active=True,
                    )

                proj = store.load_project()
                proj["last_plan"] = [
                    {"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": "p1"},
                    {"shot_index": 2, "shot_id": "shot_0002", "is_continuation": True, "prompt_sha256": "p2"},
                    {"shot_index": 3, "shot_id": "shot_0003", "is_continuation": True, "prompt_sha256": "p3"},
                ]
                store.save_project(proj)
                store.set_shot_seam_repair(2, {"enabled": True, "choice": "repaired", "mode": "rife", "rife_bridge_frames": 5})

                # Strict mode must block STALE Shot 3
                res_strict = handle_assemble_request({"project": proj_name, "stale_policy": "block_and_report"})
                self.assertFalse(res_strict["ok"])
                self.assertIn("STALE", res_strict["error"])

                # allow_with_warning exports full 3-shot master WITH Shot 2 RIFE seam repair baked in!
                res_ok = handle_assemble_request({
                    "project": proj_name,
                    "stale_policy": "allow_with_warning",
                    "output_mode": "overwrite_master",
                    "export_fcpxml": True,
                })
                self.assertTrue(res_ok["ok"], res_ok)
                self.assertTrue(os.path.isfile(res_ok["master_mp4"]))
                self.assertTrue(os.path.isfile(res_ok["fcpxml_path"]))
                self.assertEqual(res_ok["report"]["total_clips"], 3)
                self.assertEqual(len(res_ok["report"]["applied_seam_repairs"]), 1)
                self.assertGreater(res_ok["file_bytes"], 1000)

                # Decode exported MP4 with PyAV to prove video + audio are both valid
                with av.open(res_ok["master_mp4"]) as c:
                    v_cnt = sum(1 for _ in c.decode(video=0))
                    c.seek(0)
                    a_cnt = sum(1 for _ in c.decode(audio=0))
                self.assertGreaterEqual(v_cnt, 90)
                self.assertGreaterEqual(a_cnt, 20)

                # Timestamped export preserves main master alongside new version
                res_ts = handle_assemble_request({
                    "project": proj_name,
                    "stale_policy": "allow_with_warning",
                    "output_mode": "timestamped",
                    "export_fcpxml": False,
                })
                self.assertTrue(res_ts["ok"])
                self.assertNotEqual(res_ts["master_mp4"], res_ok["master_mp4"])
                st = _build_project_state_payload(proj_name)
                mp4_exports = [e for e in st["master_exports"] if e["ext"] == ".mp4"]
                self.assertEqual(len(mp4_exports), 2)
            finally:
                sr_mod.resolve_output_root = orig_resolve


if __name__ == "__main__":
    unittest.main()
