"""Phase 14 regression tests: ShotVaultWindowGuider uncapped limits, global RoPE offset, high-sigma anchor gating, zero-dup video I/O, audio_drive_mode priority, and cross-res seam repair."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

_CUSTOM_NODES_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CUSTOM_NODES_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_NODES_DIR)
_COMFY_ROOT = os.path.dirname(_CUSTOM_NODES_DIR)
if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)

_PKG_NAME = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sv_pkg = __import__(_PKG_NAME, fromlist=["NODE_CLASS_MAPPINGS"])
_cwg = __import__(f"{_PKG_NAME}.core.context_window_guider", fromlist=["*"])
_srb = __import__(f"{_PKG_NAME}.core.seam_repair_bridge", fromlist=["*"])
_vs = __import__(f"{_PKG_NAME}.core.vault_store", fromlist=["*"])
_asm = __import__(f"{_PKG_NAME}.core.assembler", fromlist=["*"])
_plan_mod = __import__(f"{_PKG_NAME}.nodes.plan_node", fromlist=["*"])


class _DummyPayloadWrap:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return _DummyPayloadWrap(cond)


class _DummyLayout:
    def __init__(self, pos_ids, segments, signature=(10, 42, 4, 4, 240)):
        self.position_ids = pos_ids
        self.segments = segments
        self.signature = signature


class TestPhase14WindowGuiderAndZeroDupIO(unittest.TestCase):
    def test_window_guider_defaults_off_and_uncapped_window_frames(self):
        wg_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]
        req = wg_cls.INPUT_TYPES()["required"]
        modes, mode_meta = req["window_mode"]
        self.assertEqual(mode_meta["default"], "off")
        for legacy in ("off", "auto_over_260f", "auto_over_192f", "force_on"):
            self.assertIn(legacy, modes)
        _, wf_meta = req["window_frames"]
        self.assertGreaterEqual(int(wf_meta["max"]), 2045)
        self.assertEqual(int(wf_meta["step"]), 17)

    def test_apply_window_rope_offset_shifts_only_target_segments_immutably(self):
        # segments: text [0:4], ref_img [4:6], cond [6:9], audio [9:13], video [13:20]
        orig_pos = torch.zeros((20, 3), dtype=torch.float64)
        orig_pos[:, 0] = torch.arange(20, dtype=torch.float64)
        orig_snapshot = orig_pos.clone()
        segs = [
            (0, 4, "text"),
            (4, 6, "ref_img"),
            (6, 9, "cond"),
            (9, 13, "audio"),
            (13, 20, "video"),
        ]
        layout = _DummyLayout(orig_pos, segs)
        conds = {
            "positive": [
                {"model_conds": {"minimax_payload": _DummyPayloadWrap({"layout": layout})}},
            ]
        }
        w1 = _cwg.Window(frame_start=221, frames=141, video_start=65, video_len=42, audio_start=368, audio_len=235)
        shifted_cnt = _cwg.apply_window_rope_offset(conds, w1)
        self.assertEqual(shifted_cnt, 1)
        # Original layout tensor must never be mutated in place
        self.assertTrue(torch.equal(layout.position_ids, orig_snapshot))
        new_layout = conds["positive"][0]["model_conds"]["minimax_payload"].cond["layout"]
        self.assertIsNot(new_layout, layout)
        new_pos = new_layout.position_ids
        # text & ref_img unchanged
        self.assertTrue(torch.equal(new_pos[0:6, 0], orig_snapshot[0:6, 0]))
        # cond & video shifted by FRAME_RESCALE * 221
        dv = _cwg.FRAME_RESCALE * 221
        self.assertTrue(torch.allclose(new_pos[6:9, 0], orig_snapshot[6:9, 0] + dv))
        self.assertTrue(torch.allclose(new_pos[13:20, 0], orig_snapshot[13:20, 0] + dv))
        # audio shifted by 368.0
        self.assertTrue(torch.allclose(new_pos[9:13, 0], orig_snapshot[9:13, 0] + 368.0))

    def test_high_sigma_gates_fake_x0_anchor_and_low_sigma_enables_it(self):
        self.assertAlmostEqual(_cwg._extract_sigma_scalar(torch.tensor([0.96])), 0.96, places=5)
        self.assertAlmostEqual(_cwg._extract_sigma_scalar(torch.tensor([960.0])), 0.96, places=5)
        self.assertAlmostEqual(_cwg._extract_sigma_scalar(torch.tensor([0.815])), 0.815, places=5)

    def test_plan_node_json_regular_mode_overrides_stale_hidden_widget(self):
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        dplan = json.dumps({
            "audio_drive_mode": "none",
            "context_frames": 22,
            "shots": [{"shot_index": 1, "duration_sec": 5.0, "shot_type": 1, "hard_cut": True, "prompt": "Test action"}],
        })
        with tempfile.TemporaryDirectory() as td:
            with patch.object(_plan_mod, "resolve_output_root", return_value=td):
                # Hidden native widget audio_drive_mode is stale "master_song_slice", master_audio_in is None
                pipe, total, _ = plan_cls.plan(
                    "StaleWidgetGuard", "auto", "", "", "16:9 (1344x768)", 0.25, 24,
                    "zero_tail_trim", "22", audio_drive_mode="master_song_slice",
                    master_audio_in=None, director_plan_json=dplan,
                )
                self.assertEqual(pipe["audio_drive_mode"], "none")
                self.assertFalse(pipe["audio_drive_active"])
                self.assertEqual(total, 1)

    def test_cross_resolution_seam_alignment_and_active_prev_priority(self):
        a = np.full((6, 672, 1216, 3), 120, dtype=np.uint8)
        b = np.full((8, 832, 1472, 3), 200, dtype=np.uint8)
        a_aligned = _srb._align_frames_u8_to_target(a, b.shape[1], b.shape[2])
        self.assertEqual(a_aligned.shape, (6, 832, 1472, 3))
        merged = np.concatenate([a_aligned, b], axis=0)
        self.assertEqual(merged.shape, (14, 832, 1472, 3))

        # Active take of shot #1 must beat stale parent_take_id when resolving prev clip
        with tempfile.TemporaryDirectory() as td:
            st = _vs.VaultStore(td)
            p_old = os.path.join(td, "shots", "shot_0001_take_001.mp4")
            p_new = os.path.join(td, "shots", "shot_0001_take_002.mp4")
            p_s2 = os.path.join(td, "shots", "shot_0002_take_001.mp4")
            os.makedirs(os.path.dirname(p_old), exist_ok=True)
            for p in (p_old, p_new, p_s2):
                with open(p, "wb") as fh:
                    fh.write(b"dummy")
            proj = st.load_project()
            proj["shots"] = {
                "1": {
                    "active_take_id": "shot_0001_take_002",
                    "takes": {
                        "shot_0001_take_001": {"video_rel_path": "shots/shot_0001_take_001.mp4"},
                        "shot_0001_take_002": {"video_rel_path": "shots/shot_0001_take_002.mp4"},
                    },
                },
                "2": {
                    "active_take_id": "shot_0002_take_001",
                    "takes": {
                        "shot_0002_take_001": {
                            "video_rel_path": "shots/shot_0002_take_001.mp4",
                            "parent_take_id": "shot_0001_take_001",
                        }
                    },
                },
            }
            st.save_project(proj)
            prev_abs, curr_abs, _, _ = _srb._resolve_prev_and_curr_paths(st, 2)
            self.assertEqual(os.path.normpath(prev_abs), os.path.normpath(p_new))
            self.assertEqual(os.path.normpath(curr_abs), os.path.normpath(p_s2))

    def test_save_6tuple_video_and_single_shot_assemble_fast_copy(self):
        save_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()
        asm_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultAssemble"]()
        self.assertEqual(len(save_cls.RETURN_TYPES), 6)
        self.assertEqual(save_cls.RETURN_TYPES[5], "VIDEO")
        self.assertEqual(len(asm_cls.RETURN_TYPES), 4)
        self.assertEqual(asm_cls.RETURN_TYPES[3], "VIDEO")

        with tempfile.TemporaryDirectory() as td:
            pipe = {
                "project_name": "FastPathDemo",
                "project_root": td,
                "fps": 24,
                "current_shot_index": 1,
                "keep_frames": 24,
                "trim_head_frames": 0,
                "overlap_frames": 0,
                "prompt": "single shot fast path",
                "prompt_sha256": "abc123",
                "seed_text": "42",
                "shot_type": 1,
                "shots": [{"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": "abc123"}],
            }
            imgs = torch.rand((24, 64, 64, 3), dtype=torch.float32)
            aud = {"waveform": torch.zeros((1, 2, 48000), dtype=torch.float32), "sample_rate": 48000}
            res = save_cls.save_take(imgs, pipe, False, 18, "h264", final_audio=aud)["result"]
            self.assertEqual(len(res), 6)
            _, _, updated_pipe, tid, vpath, vhandle = res
            self.assertTrue(os.path.isfile(vpath))
            self.assertIsNotNone(vhandle)

            # Single-shot assembly with 0 offset should use zero-reencode fast copy
            mpath, fcx, rep_s, mhandle = asm_cls.assemble(
                "FastPathDemo", "defer_until_complete", "auto_prefer_reference",
                15.0, 0.0, False, True, shot_pipe=updated_pipe,
            )["result"]
            rep = json.loads(rep_s)
            self.assertTrue(rep["single_clip_fast_copy"])
            self.assertTrue(os.path.isfile(mpath))
            self.assertTrue(os.path.isfile(fcx))
            self.assertTrue(os.path.isfile(os.path.splitext(mpath)[0] + ".manifest.json"))
            self.assertIsNotNone(mhandle)
            self.assertEqual(os.path.getsize(mpath), os.path.getsize(vpath))

            # Non-zero post_mux_offset_ms must disable fast copy and run FFmpeg mux
            _, _, rep_off_s, _ = asm_cls.assemble(
                "FastPathDemo", "defer_until_complete", "auto_prefer_reference",
                15.0, 50.0, False, False, shot_pipe=updated_pipe,
            )["result"]
            self.assertFalse(json.loads(rep_off_s)["single_clip_fast_copy"])


if __name__ == "__main__":
    unittest.main()
