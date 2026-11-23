"""Phase 12 tests: Master export manifest sync & auto-assemble after single-shot reroll,
bidirectional bridge reroll (0/1/5/22f grid/phase math, clean next-shot head loading,
reverse-taper seam-0 cleanliness, 1st+2nd pass cache, downstream VALID auto-acceptance),
and Calm UI / zero-false-alarm / modular frontend contracts.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
comfy_root = os.path.dirname(PARENT)
if comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)

SPEC = importlib.util.spec_from_file_location(
    "shotvault_pkg_p12", os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT]
)
assert SPEC and SPEC.loader
SV = importlib.util.module_from_spec(SPEC)
sys.modules["shotvault_pkg_p12"] = SV
SPEC.loader.exec_module(SV)

from shotvault_pkg_p12.core.assembler import assemble_master
from shotvault_pkg_p12.core.context_window_guider import _apply_reverse_taper_noise, build_window_guider_and_latent
from shotvault_pkg_p12.core.grid_math import compute_tail_bridge_plan
from shotvault_pkg_p12.core.lineage_dag import LineageDAG
from shotvault_pkg_p12.core.script_parser import parse_director_plan_json
from shotvault_pkg_p12.core.seam_qc import write_mp4_with_pcm
from shotvault_pkg_p12.core.vault_store import VaultStore, compute_master_sync_status, load_next_shot_head_ref


class _DummyVAE:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, img):
        self.encode_calls += 1
        f = int(img.shape[0])
        lt = 1 if f <= 1 else (2 if f <= 5 else ((f - 5) // 17) * 5 + 2)
        h = max(2, int(img.shape[1]) // 16)
        w = max(2, int(img.shape[2]) // 16)
        return torch.zeros((1, 24, lt, h, w), dtype=torch.float32)


class _DummyAudioVAE:
    audio_sample_rate = 32000

    def encode(self, wav):
        t = max(1, int(wav.shape[1]) // 800)
        return torch.zeros((1, 16, 2, t), dtype=torch.float32)


class TestPhase12BidirRerollExportSyncAndUX(unittest.TestCase):
    def test_tail_bridge_grid_math_regular_and_audio_driven(self):
        # Regular mode: trim_head=22, orig_keep=204 -> base=226 (17*13+5)
        p0 = compute_tail_bridge_plan(22, 204, 0, audio_drive_active=False)
        self.assertEqual(p0["target_frames"], 226)
        self.assertEqual(p0["keep_frames"], 204)
        self.assertEqual(p0["tail_bridge_frames"], 0)

        p1 = compute_tail_bridge_plan(22, 204, 1, audio_drive_active=False)
        self.assertEqual((p1["target_frames"] - 5) % 17, 0)
        self.assertEqual(p1["tail_anchor_index"], p1["target_frames"] - 1)
        self.assertEqual(p1["keep_frames"], p1["tail_anchor_index"] - 22)

        for g in (5, 22):
            p = compute_tail_bridge_plan(22, 204, g, audio_drive_active=False)
            self.assertEqual((p["target_frames"] - 5) % 17, 0)
            self.assertEqual(p["tail_anchor_index"] % 17, 0)  # strict cycle_pos == 0!
            self.assertEqual(p["tail_anchor_index"] + g, p["target_frames"])
            self.assertEqual(p["keep_frames"], p["tail_anchor_index"] - 22)

        # Audio-driven / MV mode: strict 0-frame drift (keep_frames === 204)
        for g in (1, 5, 22):
            pm = compute_tail_bridge_plan(39, 204, g, audio_drive_active=True)
            self.assertEqual((pm["target_frames"] - 5) % 17, 0)
            self.assertEqual(pm["keep_frames"], 204)
            self.assertEqual(pm["tail_anchor_index"], 39 + 204)
            self.assertLessEqual(pm["tail_anchor_index"] + g, pm["target_frames"])

    def test_reverse_taper_noise_keeps_seam_frame_0_clean_and_tapers_away(self):
        clean = torch.full((22, 64, 96, 3), 0.45, dtype=torch.float32)
        noisy = _apply_reverse_taper_noise(clean, alpha=0.25, ramp_frames=4, seed=12345)
        # Position 0 touches the seam -> MUST be bit-exact 0.0 difference!
        self.assertTrue(torch.equal(noisy[0], clean[0]))
        # Positions 1..3 ramp up gently; positions >= 4 reach mild alpha=0.25
        err_1 = (noisy[1] - clean[1]).abs().mean().item()
        err_10 = (noisy[10] - clean[10]).abs().mean().item()
        self.assertGreater(err_1, 0.0)
        self.assertGreater(err_10, err_1)

    def test_window_guider_bidir_tail_injection_and_2nd_pass_cache(self):
        import comfy.nested_tensor as nt

        vae = _DummyVAE()
        avae = _DummyAudioVAE()
        v_lat = torch.zeros((1, 24, 72, 8, 12), dtype=torch.float32)  # 17*14+5 = 243f
        a_lat = torch.zeros((1, 16, 2, 405), dtype=torch.float32)
        latent = {"samples": nt.NestedTensor((v_lat, a_lat))}
        pos = [[torch.zeros((1, 8, 64)), {}]]
        prev_img = torch.full((22, 128, 192, 3), 0.4, dtype=torch.float32)
        next_img = torch.full((22, 128, 192, 3), 0.6, dtype=torch.float32)
        next_aud = {"waveform": torch.full((1, 2, 26667), 0.05, dtype=torch.float32), "sample_rate": 32000}

        pipe = {
            "current_shot_index": 5,
            "parent_take_id": "shot_0004_take_001",
            "is_continuation": True,
            "overlap_frames": 22,
            "context_frames": 22,
            "tail_bridge_frames": 22,
            "tail_bridge_index": 221,
            "tail_bridge_noise": "reverse_taper",
            "bridged_next_take_id": "shot_0006_take_001",
            "next_head_clean_ref": next_img,
            "next_head_audio_32k": next_aud,
        }

        # 1st pass: encodes head + tail video
        res1, _ = build_window_guider_and_latent(
            model=None, positive=pos, latent=latent, vae=vae, audio_vae=avae,
            tapered_prev_images=prev_img, shot_pipe=pipe, pass_stage="single_or_1st", continuation_frames="auto",
        )
        kfs1 = res1[0][1]["minimax_keyframes"]
        v_kfs1 = [k for k in kfs1 if "latent" in k]
        a_kfs1 = [k for k in kfs1 if k.get("shotvault_tail_bridge") and "audio_latent" in k]
        self.assertEqual([k["resolved_frame_index"] for k in v_kfs1], [0, 221])
        self.assertEqual(len(a_kfs1), 1)
        self.assertEqual(a_kfs1[0]["resolved_frame_index"], 223)
        calls_after_1st = vae.encode_calls
        self.assertEqual(calls_after_1st, 2)

        # 2nd pass: MUST reuse both head and tail cached latents (zero extra VAE encodes!)
        res2, _ = build_window_guider_and_latent(
            model=None, positive=pos, latent=latent, vae=vae, audio_vae=avae,
            tapered_prev_images=prev_img, shot_pipe=pipe, pass_stage="2nd_refine", continuation_frames="auto",
        )
        self.assertEqual(vae.encode_calls, calls_after_1st)
        v_kfs2 = [k for k in res2[0][1]["minimax_keyframes"] if "latent" in k]
        self.assertEqual([k["resolved_frame_index"] for k in v_kfs2], [0, 221])

    def test_next_head_load_manifest_sync_downstream_valid_and_auto_assemble(self):
        with tempfile.TemporaryDirectory() as td:
            store = VaultStore(td)
            img1 = torch.full((36, 128, 192, 3), 0.30, dtype=torch.float32)
            img2 = torch.full((36, 128, 192, 3), 0.50, dtype=torch.float32)
            img3 = torch.full((36, 128, 192, 3), 0.70, dtype=torch.float32)
            aud = torch.zeros((2, int(round(36 / 24.0 * 48000))), dtype=torch.float32)

            for idx, (tid, img, parent) in enumerate([
                ("shot_0001_take_001", img1, ""),
                ("shot_0002_take_001", img2, "shot_0001_take_001"),
                ("shot_0003_take_001", img3, "shot_0002_take_001"),
            ], start=1):
                vrel = f"shots/{tid}.mp4"
                write_mp4_with_pcm(img, aud, 48000, 24, os.path.join(td, vrel))
                store.save_take_context(
                    take_id=tid, shot_index=idx, clean_images=img, audio_waveform_48k=aud,
                    audio_sr=48000, fps=24, prompt_sha256=f"p{idx}", parent_take_id=parent,
                    seed_text=str(idx), keep_frames=36, trim_head_frames=0, shot_type=1,
                    video_rel_path=vrel, set_active=True,
                )

            plan = [
                {"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": "p1"},
                {"shot_index": 2, "shot_id": "shot_0002", "is_continuation": True, "prompt_sha256": "p2"},
                {"shot_index": 3, "shot_id": "shot_0003", "is_continuation": True, "prompt_sha256": "p3"},
            ]
            proj = store.load_project()
            proj["last_plan"] = plan
            store.save_project(proj)

            # 1. load_next_shot_head_ref decodes exact clean 5f & 22f head from shot 3 MP4
            ref5 = load_next_shot_head_ref(store, 3, 5, fps=24)
            self.assertIsNotNone(ref5)
            self.assertEqual(tuple(ref5["images"].shape), (5, 128, 192, 3))
            self.assertAlmostEqual(float(ref5["images"].mean().item()), 0.70, delta=0.05)

            ref22 = load_next_shot_head_ref(store, 3, 22, fps=24)
            self.assertIsNotNone(ref22)
            self.assertEqual(tuple(ref22["images"].shape), (22, 128, 192, 3))

            # 2. Before exporting master -> is_outdated is True
            st0 = compute_master_sync_status(store, plan)
            self.assertFalse(st0["has_master"])
            self.assertTrue(st0["is_outdated"])

            # 3. Export initial master -> writes .manifest.json and is_outdated becomes False
            m_mp4, _, _ = assemble_master(store, plan, fps=24, stale_policy="allow_with_warning", export_fcpxml=True)
            self.assertTrue(os.path.isfile(m_mp4))
            st1 = compute_master_sync_status(store, plan)
            self.assertTrue(st1["has_master"])
            self.assertFalse(st1["is_outdated"])

            # 4. Run ShotVaultSave for rerolled Shot 2 WITH bridged_next_take_id + auto_assemble_after_reroll=True
            save_node = SV.NODE_CLASS_MAPPINGS["ShotVaultSave"]()
            pipe_reroll = {
                "project_root": td,
                "project_name": os.path.basename(td),
                "current_shot_index": 2,
                "fps": 24,
                "prompt_sha256": "p2",
                "parent_take_id": "shot_0001_take_001",
                "seed_text": "999",
                "keep_frames": 36,
                "trim_head_frames": 0,
                "shot_type": 1,
                "bridged_next_take_id": "shot_0003_take_001",
                "tail_bridge_frames": 5,
                "shots": plan,
                "retry_lock": {"enabled": True, "shot_index": 2, "tail_bridge_frames": 5, "auto_assemble_after_reroll": True},
            }
            img2_new = torch.full((36, 128, 192, 3), 0.55, dtype=torch.float32)
            out_save = save_node.save_take(
                final_images=img2_new,
                shot_pipe=pipe_reroll,
                auto_queue_next=False,
                crf=20,
                codec="h264",
                final_audio={"waveform": aud.unsqueeze(0), "sample_rate": 48000},
            )
            updated_pipe = out_save["result"][2]
            new_tid = out_save["result"][3]
            self.assertEqual(new_tid, "shot_0002_take_002")
            # Auto-assembled master is fresh & includes shot_0002_take_002!
            self.assertTrue(os.path.isfile(updated_pipe.get("auto_assembled_master", "")))
            st2 = compute_master_sync_status(store, plan)
            self.assertFalse(st2["is_outdated"])

            # Downstream Shot 3 must remain green VALID because Shot 2 bridged onto shot_0003_take_001!
            dag = LineageDAG.evaluate(plan, store.load_project())
            self.assertEqual([d["status"] for d in dag], ["VALID", "VALID", "VALID"])

            # Switching Shot 2 back to take_001 without re-exporting -> is_outdated immediately flags Shot 2!
            store.set_active_take(2, "shot_0002_take_001")
            st3 = compute_master_sync_status(store, plan)
            self.assertTrue(st3["is_outdated"])
            self.assertIn(2, st3["outdated_shots"])

    def test_v58_modular_frontend_calm_ui_and_retry_lock_parsing(self):
        # Verify Calm UI: NO heuristic '⚠连' badges in director_panel.js, modular panel/* files exist
        dp = open(os.path.join(ROOT, "web", "director_panel.js"), "r", encoding="utf-8").read()
        self.assertNotIn("⚠连", dp)
        self.assertIn("appendSafe", dp)
        self.assertIn("extractCardSnippet", dp)
        self.assertIn("makeDraggableTextarea", dp)
        self.assertIn("openFocusPromptModal", dp)
        self.assertIn("node.onResize", dp)
        for mod_fn in ("styles.js", "prompt_modal.js", "export_modal.js"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, "web", "panel", mod_fn)))

        # Verify retry_lock v5.8 fields parse cleanly
        gcfg, _ = parse_director_plan_json(
            json.dumps({
                "retry_lock": {
                    "enabled": True,
                    "shot_index": 5,
                    "tail_bridge_frames": 5,
                    "tail_bridge_noise": "reverse_taper",
                    "auto_assemble_after_reroll": False,
                },
                "shots": [{"duration_sec": 5.0, "prompt": "a"}],
            }),
            common_prompt="",
            fps=24,
        )
        rl = gcfg["retry_lock"]
        self.assertEqual(rl["tail_bridge_frames"], 5)
        self.assertEqual(rl["tail_bridge_noise"], "reverse_taper")
        self.assertFalse(rl["auto_assemble_after_reroll"])


if __name__ == "__main__":
    unittest.main()
