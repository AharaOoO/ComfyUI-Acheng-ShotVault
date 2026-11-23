"""Phase 10 tests: Seam-window QC isolation, bounded once_ladder retry, manual enable_repair,
on-demand single-seam FuzzPuppy preview + assembler integration, and single/multi/all cleanup + recycle restore.
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

# Import ShotVault package cleanly regardless of folder hyphen
SPEC = importlib.util.spec_from_file_location(
    "shotvault_pkg", os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT]
)
assert SPEC and SPEC.loader
SV = importlib.util.module_from_spec(SPEC)
sys.modules["shotvault_pkg"] = SV
SPEC.loader.exec_module(SV)

from shotvault_pkg.core.assembler import assemble_master
from shotvault_pkg.core.seam_qc import analyze_seam_diagnostics, write_mp4_with_pcm
from shotvault_pkg.core.seam_repair_bridge import repair_shot_seam_preview
from shotvault_pkg.core.vault_store import VaultStore


def _load_seam_bridge():
    sb_dir = os.path.join(PARENT, "ComfyUI-Acheng-SeamBridge")
    spec = importlib.util.spec_from_file_location(
        "acheng_sb_pkg", os.path.join(sb_dir, "__init__.py"), submodule_search_locations=[sb_dir]
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["acheng_sb_pkg"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestPhase10SeamRepairAndCleanup(unittest.TestCase):
    def test_seam_qc_isolates_interior_flash_from_true_seam(self):
        # Smooth seam at join_frame=24, with bright interior flash at f66 (+42f after seam)
        prev = torch.full((24, 64, 96, 3), 0.35, dtype=torch.float32)
        curr = torch.full((48, 64, 96, 3), 0.35, dtype=torch.float32)
        curr[42:, ...] = 0.85  # f66 interior flash
        diag = analyze_seam_diagnostics(prev, curr, fps=24)
        self.assertEqual(diag["status"], "ok")
        self.assertEqual(diag["events"], [])
        self.assertTrue(len(diag["interior_events"]) >= 1)
        self.assertEqual(diag["interior_events"][0]["frame"], 66)

        # True seam jump right at f24 MUST still trigger warning
        curr_bad = torch.full((48, 64, 96, 3), 0.85, dtype=torch.float32)
        diag_bad = analyze_seam_diagnostics(prev, curr_bad, fps=24)
        self.assertEqual(diag_bad["status"], "warning")
        self.assertTrue(any(ev["frame"] == 24 for ev in diag_bad["events"]))

    def test_fuzzpuppy_retry_policy_off_and_once_ladder_and_enable_switch(self):
        sb = _load_seam_bridge()
        from acheng_sb_pkg.fuzzpuppy_engine import merge_clips_fuzzpuppy

        rng = np.random.default_rng(123)
        base = rng.uniform(50, 200, (1, 128, 192, 3)).astype(np.uint8)
        fa = np.repeat(base, 32, axis=0)
        fb = np.clip(base.astype(np.int32) + 8, 0, 255).astype(np.uint8)
        fb = np.repeat(fb, 32, axis=0)

        out_off, _, rep_off = merge_clips_fuzzpuppy(fa, fb, fps=24.0, mode="rife", rife_bridge_frames=5, retry_policy="off")
        self.assertIn("seam_qc", rep_off)
        self.assertIn("summary_zh", rep_off)
        self.assertIn("summary_en", rep_off)
        self.assertEqual(rep_off["retry_policy"], "off")
        self.assertNotIn("ladder_trials", rep_off)
        self.assertEqual(len(out_off), 60)

        # Verify ClipMerge schema has enable_repair=False, retry_policy='off', ui_lang='zh', shot_pipe appended cleanly at end
        inp = sb.NODE_CLASS_MAPPINGS["AchengFuzzPuppyClipMerge"].INPUT_TYPES()
        opt_keys = list(inp["optional"].keys())
        self.assertEqual(opt_keys, ["is_continuation", "codec", "enable_repair", "retry_policy", "ui_lang", "shot_pipe"])
        self.assertEqual(inp["optional"]["enable_repair"][1]["default"], False)
        self.assertEqual(inp["optional"]["retry_policy"][1]["default"], "off")

    def test_history_cleanup_single_multi_all_protection_and_recycle_restore(self):
        with tempfile.TemporaryDirectory() as td:
            store = VaultStore(td)
            img = torch.full((12, 64, 96, 3), 0.4, dtype=torch.float32)
            aud = torch.zeros((2, 24000), dtype=torch.float32)

            def _add_take(shot_idx, tid, parent="", set_active=True):
                vrel = f"shots/{tid}.mp4"
                trel = f"_vault/thumb_{tid}.jpg"
                srel = f"_vault/seam_{tid}.mp4"
                os.makedirs(os.path.join(td, "shots"), exist_ok=True)
                os.makedirs(os.path.join(td, "_vault"), exist_ok=True)
                with open(os.path.join(td, vrel), "wb") as f:
                    f.write(b"0" * 4096)
                with open(os.path.join(td, trel), "wb") as f:
                    f.write(b"1" * 1024)
                with open(os.path.join(td, srel), "wb") as f:
                    f.write(b"2" * 2048)
                store.save_take_context(
                    take_id=tid, shot_index=shot_idx, clean_images=img, audio_waveform_48k=aud,
                    audio_sr=48000, fps=24, prompt_sha256="abc", parent_take_id=parent, seed_text="1",
                    keep_frames=12, trim_head_frames=0, shot_type=1, video_rel_path=vrel,
                    thumb_rel_path=trel, seam_rel_path=srel, set_active=set_active,
                )

            _add_take(1, "shot_0001_take_001", set_active=True)   # will be parent of shot 2!
            _add_take(1, "shot_0001_take_002", set_active=False)  # unused -> deletable
            _add_take(1, "shot_0001_take_003", set_active=True)   # active -> protected
            _add_take(2, "shot_0002_take_001", parent="shot_0001_take_001", set_active=True)
            _add_take(2, "shot_0002_take_002", set_active=False)  # unused -> deletable

            inv = store.get_history_inventory()
            self.assertEqual(inv["unused_count"], 2)
            s1_takes = {t["take_id"]: t for t in inv["shots"][0]["takes"]}
            self.assertIn("parent_ref", s1_takes["shot_0001_take_001"]["protected_reasons"])
            self.assertFalse(s1_takes["shot_0001_take_001"]["deletable"])
            self.assertTrue(s1_takes["shot_0001_take_002"]["deletable"])
            self.assertIn("active", s1_takes["shot_0001_take_003"]["protected_reasons"])

            # Attempt deleting a protected take -> must be skipped!
            res_prot = store.cleanup_items(take_ids=["shot_0001_take_001", "shot_0001_take_003"], scope="selected", mode="recycle")
            self.assertEqual(res_prot["deleted_takes"], [])
            self.assertEqual(len(res_prot["skipped_protected"]), 2)

            # Single delete unused take -> moves to recycle bin AND removes from project.json
            res_one = store.cleanup_items(take_ids=["shot_0001_take_002"], scope="selected", mode="recycle")
            self.assertEqual(res_one["deleted_takes"], ["shot_0001_take_002"])
            self.assertTrue(res_one["batch_id"])
            self.assertFalse(os.path.exists(os.path.join(td, "shots/shot_0001_take_002.mp4")))
            self.assertNotIn("shot_0001_take_002", store.load_project()["shots"]["1"]["takes"])

            # Restore from recycle bin -> file & project.json record both come back!
            res_rest = store.manage_recycle_bin(action="restore", batch_ids=[res_one["batch_id"]])
            self.assertEqual(res_rest["restored_takes"], ["shot_0001_take_002"])
            self.assertTrue(os.path.exists(os.path.join(td, "shots/shot_0001_take_002.mp4")))
            self.assertIn("shot_0001_take_002", store.load_project()["shots"]["1"]["takes"])

            # Select-all project_unused + empty recycle bin
            res_all = store.cleanup_items(scope="project_unused", mode="recycle")
            self.assertEqual(sorted(res_all["deleted_takes"]), ["shot_0001_take_002", "shot_0002_take_002"])
            res_empty = store.manage_recycle_bin(action="empty")
            self.assertGreater(res_empty["freed_bytes"], 0)
            self.assertEqual(store.manage_recycle_bin(action="list")["batches"], [])

    def test_single_seam_repair_preview_and_assembler_toggle(self):
        with tempfile.TemporaryDirectory() as td:
            store = VaultStore(td)
            rng = np.random.default_rng(77)
            base = torch.from_numpy(rng.uniform(0.25, 0.75, (1, 128, 192, 3)).astype(np.float32))
            img1 = base.repeat(36, 1, 1, 1)
            img2 = torch.clamp(base.repeat(36, 1, 1, 1) + 0.03, 0.0, 1.0)
            aud = torch.zeros((2, int(36 / 24.0 * 48000)), dtype=torch.float32)

            p1 = os.path.join(td, "shots/shot_0001_take_001.mp4")
            p2 = os.path.join(td, "shots/shot_0002_take_001.mp4")
            write_mp4_with_pcm(img1, aud, 48000, 24, p1)
            write_mp4_with_pcm(img2, aud, 48000, 24, p2)

            store.save_take_context(
                take_id="shot_0001_take_001", shot_index=1, clean_images=img1, audio_waveform_48k=aud,
                audio_sr=48000, fps=24, prompt_sha256="p1", parent_take_id="", seed_text="1",
                keep_frames=36, trim_head_frames=0, shot_type=1, video_rel_path="shots/shot_0001_take_001.mp4",
                set_active=True,
            )
            store.save_take_context(
                take_id="shot_0002_take_001", shot_index=2, clean_images=img2, audio_waveform_48k=aud,
                audio_sr=48000, fps=24, prompt_sha256="p2", parent_take_id="shot_0001_take_001", seed_text="2",
                keep_frames=36, trim_head_frames=0, shot_type=1, video_rel_path="shots/shot_0002_take_001.mp4",
                set_active=True,
            )

            # On-demand 3s seam repair preview
            rep = repair_shot_seam_preview(store, shot_index=2, mode="rife", rife_bridge_frames=5, retry_policy="off")
            self.assertTrue(rep["ok"])
            repaired_abs = os.path.join(td, rep["seam_repair"]["repaired_seam_rel"])
            self.assertTrue(os.path.isfile(repaired_abs))
            self.assertIn("raw_seam_qc", rep["seam_repair"]["report"])

            plan = [
                {"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": "p1"},
                {"shot_index": 2, "shot_id": "shot_0002", "is_continuation": True, "prompt_sha256": "p2"},
            ]

            # Assemble WITH repair enabled -> applies seam repair!
            m_mp4, _, m_rep_json = assemble_master(store, plan, fps=24, stale_policy="allow_with_warning", export_fcpxml=False, master_filename="master_repaired")
            m_rep = json.loads(m_rep_json)
            self.assertTrue(os.path.isfile(m_mp4))
            self.assertEqual(len(m_rep["applied_seam_repairs"]), 1)

            # Toggle shot 2 repair OFF -> assemble keeps original untouched!
            store.set_shot_seam_repair(2, {"enabled": False, "choice": "original"})
            m2_mp4, _, m2_rep_json = assemble_master(store, plan, fps=24, stale_policy="allow_with_warning", export_fcpxml=False, master_filename="master_orig")
            m2_rep = json.loads(m2_rep_json)
            self.assertTrue(os.path.isfile(m2_mp4))
            self.assertEqual(m2_rep["applied_seam_repairs"], [])

    def test_clip_merge_shot_pipe_sync_and_preview_playback(self):
        import av
        from fractions import Fraction

        # Add ComfyUI root so comfy_api.latest can be imported
        comfy_root = os.path.dirname(PARENT)
        if comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)
        from comfy_api.latest import InputImpl, Types

        sb = _load_seam_bridge()
        node = sb.NODE_CLASS_MAPPINGS["AchengFuzzPuppyClipMerge"]()

        with tempfile.TemporaryDirectory() as td:
            store = VaultStore(td)
            rng = np.random.default_rng(99)
            base = torch.from_numpy(rng.uniform(0.25, 0.75, (1, 128, 192, 3)).astype(np.float32))
            img1 = base.repeat(36, 1, 1, 1)
            img2 = torch.clamp(base.repeat(36, 1, 1, 1) + 0.02, 0.0, 1.0)
            # Non-silent stereo 48kHz tone so AAC encoding & decoding are genuinely tested
            n_samps = int(round(36 / 24.0 * 48000))
            t_vec = torch.linspace(0.0, 1.5, n_samps, dtype=torch.float32)
            tone = 0.1 * torch.sin(2.0 * np.pi * 440.0 * t_vec)
            aud = torch.stack([tone, tone], dim=0)

            p1 = os.path.join(td, "shots/shot_0001_take_001.mp4")
            p2 = os.path.join(td, "shots/shot_0002_take_001.mp4")
            write_mp4_with_pcm(img1, aud, 48000, 24, p1)
            write_mp4_with_pcm(img2, aud, 48000, 24, p2)

            store.save_take_context(
                take_id="shot_0001_take_001", shot_index=1, clean_images=img1, audio_waveform_48k=aud,
                audio_sr=48000, fps=24, prompt_sha256="p1", parent_take_id="", seed_text="1",
                keep_frames=36, trim_head_frames=0, shot_type=1, video_rel_path="shots/shot_0001_take_001.mp4",
                set_active=True,
            )
            store.save_take_context(
                take_id="shot_0002_take_001", shot_index=2, clean_images=img2, audio_waveform_48k=aud,
                audio_sr=48000, fps=24, prompt_sha256="p2", parent_take_id="shot_0001_take_001", seed_text="2",
                keep_frames=36, trim_head_frames=0, shot_type=1, video_rel_path="shots/shot_0002_take_001.mp4",
                set_active=True,
            )

            vid_a = InputImpl.VideoFromComponents(
                Types.VideoComponents(images=img1, audio={"waveform": aud.unsqueeze(0), "sample_rate": 48000}, frame_rate=Fraction(24, 1))
            )
            vid_b = InputImpl.VideoFromComponents(
                Types.VideoComponents(images=img2, audio={"waveform": aud.unsqueeze(0), "sample_rate": 48000}, frame_rate=Fraction(24, 1))
            )
            pipe = {
                "project_root": td,
                "current_shot_index": 2,
                "last_saved_take_id": "shot_0002_take_001",
            }

            # Case A: Default enable_repair=False + no override -> lossless original, 0 reencode
            _, diag_a = node.merge(vid_a, vid_b, enable_repair=False, shot_pipe=pipe)
            rep_a = json.loads(diag_a)
            self.assertEqual(rep_a["applied"], "manual_disabled_original")
            self.assertEqual(rep_a["reencoded_chunks"], 0)
            self.assertEqual(rep_a["control_source"], "node_global")

            # Case B: enable_repair=False + Director Console override enabled=True/auto_align -> runs auto_align & syncs preview
            store.set_shot_seam_repair(2, {"enabled": True, "choice": "repaired", "mode": "auto_align"})
            _, diag_b = node.merge(vid_a, vid_b, enable_repair=False, shot_pipe=pipe)
            rep_b = json.loads(diag_b)
            self.assertEqual(rep_b["control_source"], "director_console_per_shot")
            self.assertEqual(rep_b["mode"], "auto_align")
            self.assertTrue(rep_b["safe"])
            synced_rel_b = store.load_project()["shots"]["2"]["seam_repair"]["repaired_seam_rel"]
            self.assertTrue(os.path.isfile(os.path.join(td, synced_rel_b)))

            # Case C: enable_repair=True + Director Console override enabled=False/original -> skips repair!
            store.set_shot_seam_repair(2, {"enabled": False, "choice": "original"})
            _, diag_c = node.merge(vid_a, vid_b, enable_repair=True, shot_pipe=pipe)
            rep_c = json.loads(diag_c)
            self.assertEqual(rep_c["applied"], "manual_disabled_original")
            self.assertEqual(rep_c["control_source"], "director_console_per_shot")
            self.assertEqual(rep_c["reencoded_chunks"], 0)

            # Case D: enable_repair=True + no override -> runs rife, syncs project.json + valid playable MP4 (video + AAC audio!)
            proj_d = store.load_project()
            proj_d["shots"]["2"].pop("seam_repair", None)
            store.save_project(proj_d)
            _, diag_d = node.merge(vid_a, vid_b, mode="rife", rife_bridge_frames="5", enable_repair=True, shot_pipe=pipe)
            rep_d = json.loads(diag_d)
            self.assertTrue(rep_d["safe"])
            self.assertEqual(rep_d["control_source"], "node_global")
            sr_d = store.load_project()["shots"]["2"]["seam_repair"]
            self.assertTrue(sr_d["enabled"])
            self.assertEqual(sr_d["choice"], "repaired")
            self.assertEqual(sr_d["take_id"], "shot_0002_take_001")
            repaired_mp4 = os.path.join(td, sr_d["repaired_seam_rel"])
            self.assertTrue(os.path.isfile(repaired_mp4))
            with av.open(repaired_mp4) as container:
                v_frames = list(container.decode(video=0))
                container.seek(0)
                a_frames = list(container.decode(audio=0))
            self.assertGreaterEqual(len(v_frames), 10)
            self.assertGreaterEqual(len(a_frames), 10)


if __name__ == "__main__":
    unittest.main()
