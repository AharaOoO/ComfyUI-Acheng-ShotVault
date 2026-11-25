"""Phase 15: 3-slot reference voice reuse, decoupled inheritance, sparse <Audio k> remapping,
audio-drive mute invariance, IS_CHANGED cache invalidation, and frontend UI/i18n verification."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "sv_p15_pkg"
if PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(PKG_NAME)
    _pkg.__path__ = [REPO_ROOT]
    _pkg.__package__ = PKG_NAME
    sys.modules[PKG_NAME] = _pkg

from sv_p15_pkg.core import asset_and_turnaround as aat  # type: ignore
from sv_p15_pkg.core.script_parser import parse_director_plan_json  # type: ignore
from sv_p15_pkg.nodes.load_node import ShotVaultLoad  # type: ignore
from sv_p15_pkg.nodes.loop_dispatch_node import ShotVaultLoopDispatch  # type: ignore

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _fake_audio_loader(path: str):
    tag = float(sum(ord(c) for c in os.path.basename(path)) % 97 + 1)
    wf = torch.full((1, 1, 1600), tag, dtype=torch.float32)
    return {"waveform": wf, "sample_rate": 16000, "_tag": os.path.basename(path)}


def _fake_image_loader(path: str):
    tag = float(sum(ord(c) for c in os.path.basename(path)) % 97 + 1)
    return torch.full((1, 16, 16, 3), tag, dtype=torch.float32)


class TestPhase15VoiceRefReuseAndSlots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for fn in ["v1.wav", "v2.wav", "v2_alt.wav", "v3.wav", "g1.wav", "g2.wav", "own1.wav", "p1.png", "p2.png"]:
            with open(os.path.join(self.root, fn), "wb") as f:
                f.write(b"RIFF0000WAVEfmt ")

    def tearDown(self):
        self.tmp.cleanup()

    def test_t1_3slot_override_and_decoupled_inherit(self):
        plan_dict = {
            "global_ref_slots": ["p1.png"] + [""] * 8,
            "global_ref_audio_slots": ["v1.wav", "v2.wav", ""],
            "shots": [
                {
                    "shot_index": 1,
                    "duration_sec": 5.0,
                    "prompt": "<Audio 1> says hi, <Audio 2> replies, <Audio 3> laughs",
                    "inherit_global_refs": False,
                    "inherit_global_audios": True,
                    "ref_slots": [""] * 9,
                    "ref_audio_slots": ["", "v2_alt.wav", "v3.wav"],
                },
                {
                    "shot_index": 2,
                    "duration_sec": 5.0,
                    "prompt": "<Audio 2> and <Audio 3>",
                    "inherit_global_refs": True,
                    "inherit_global_audios": False,
                    "ref_slots": [""] * 9,
                    "ref_audio_slots": ["", "v2_alt.wav", "v3.wav"],
                },
            ],
        }
        meta, shots = parse_director_plan_json(json.dumps(plan_dict), fps=24)
        self.assertEqual(len(shots), 2)
        self.assertEqual(meta["global_ref_audio_slots"], ["v1.wav", "v2.wav", ""])
        self.assertEqual(meta["global_ref_audios"], ["v1.wav", "v2.wav"])

        s1, s2 = shots
        self.assertFalse(s1["inherit_global_refs"])
        self.assertTrue(s1["inherit_global_audios"])
        self.assertTrue(s2["inherit_global_refs"])
        self.assertFalse(s2["inherit_global_audios"])

        gcfg = {
            "global_ref_slots": meta["global_ref_slots"],
            "global_ref_images": meta["global_ref_images"],
            "global_ref_audio_slots": meta["global_ref_audio_slots"],
            "global_ref_audios": meta["global_ref_audios"],
        }
        with patch.object(aat, "_load_audio_dict", side_effect=_fake_audio_loader), \
             patch.object(aat, "_load_image_tensor", side_effect=_fake_image_loader):
            mm1 = aat.load_shot_multimodal_refs(
                s1,
                global_cfg=gcfg,
                input_dir=self.root,
                skip_ref_audios=False,
            )
            # Images did NOT inherit (inherit_global_refs=False), voices DID inherit slot 1
            self.assertEqual(len(mm1["images"]), 0)
            self.assertEqual([a["_tag"] for a in mm1["audios"]], ["v1.wav", "v2_alt.wav", "v3.wav"])
            self.assertEqual(mm1["audio_slot_to_ordinal"], {1: 1, 2: 2, 3: 3})

            mm2 = aat.load_shot_multimodal_refs(
                s2,
                global_cfg=gcfg,
                input_dir=self.root,
                skip_ref_audios=False,
            )
            # Images DID inherit (inherit_global_refs=True), voices did NOT inherit slot 1
            self.assertEqual(len(mm2["images"]), 1)
            self.assertEqual([a["_tag"] for a in mm2["audios"]], ["v2_alt.wav", "v3.wav"])
            self.assertEqual(mm2["audio_slot_to_ordinal"], {2: 1, 3: 2})
            self.assertIsNone(mm2["slot_audios"][0])
            self.assertEqual(mm2["slot_audios"][1]["_tag"], "v2_alt.wav")
            self.assertEqual(mm2["slot_audios"][2]["_tag"], "v3.wav")

    def test_t2_legacy_plan_json_backwards_compat(self):
        legacy = {
            "global_ref_audios": ["g1.wav", "g2.wav"],
            "shots": [
                {
                    "shot_index": 1,
                    "duration_sec": 5.0,
                    "prompt": "legacy inherit true, empty own",
                    "inherit_global_refs": True,
                    "ref_audios": [],
                },
                {
                    "shot_index": 2,
                    "duration_sec": 5.0,
                    "prompt": "legacy inherit false, own1 only",
                    "inherit_global_refs": False,
                    "ref_audios": ["own1.wav"],
                },
                {
                    "shot_index": 3,
                    "duration_sec": 5.0,
                    "prompt": "legacy inherit true, preserves pre-v5.8.4 global+own append order",
                    "inherit_global_refs": True,
                    "ref_audios": ["own1.wav"],
                },
            ],
        }
        meta, shots = parse_director_plan_json(json.dumps(legacy), fps=24)
        self.assertEqual(meta["global_ref_audios"], ["g1.wav", "g2.wav"])
        s1, s2, s3 = shots
        self.assertTrue(s1["inherit_global_audios"])
        self.assertFalse(s2["inherit_global_audios"])
        self.assertTrue(s3["inherit_global_audios"])

        gcfg = {"global_ref_audios": meta["global_ref_audios"]}
        with patch.object(aat, "_load_audio_dict", side_effect=_fake_audio_loader):
            mm1 = aat.load_shot_multimodal_refs(s1, global_cfg=gcfg, input_dir=self.root)
            self.assertEqual([a["_tag"] for a in mm1["audios"]], ["g1.wav", "g2.wav"])

            mm2 = aat.load_shot_multimodal_refs(s2, global_cfg=gcfg, input_dir=self.root)
            self.assertEqual([a["_tag"] for a in mm2["audios"]], ["own1.wav"])

            # Pure legacy JSON without ref_audio_slots/global_ref_audio_slots preserves pre-v5.8.4 append order
            mm3 = aat.load_shot_multimodal_refs(s3, global_cfg=gcfg, input_dir=self.root)
            self.assertEqual([a["_tag"] for a in mm3["audios"]], ["g1.wav", "g2.wav", "own1.wav"])
            self.assertEqual(mm3["audio_slot_to_ordinal"], {1: 1, 2: 2, 3: 3})

            # Once upgraded with global_ref_audio_slots + ref_audio_slots, slot 1 overrides g1 while slot 2 inherits g2
            s3_upgraded = dict(s3, ref_audio_slots=["own1.wav", "", ""])
            gcfg_upgraded = {"global_ref_audio_slots": ["g1.wav", "g2.wav", ""], "global_ref_audios": ["g1.wav", "g2.wav"]}
            mm3_up = aat.load_shot_multimodal_refs(s3_upgraded, global_cfg=gcfg_upgraded, input_dir=self.root)
            self.assertEqual([a["_tag"] for a in mm3_up["audios"]], ["own1.wav", "g2.wav"])
            self.assertEqual(mm3_up["audio_slot_to_ordinal"], {1: 1, 2: 2})

    def test_t3_sparse_audio_tag_remap_and_video_audio_safety(self):
        # Verify the exact remapping helper behavior in h3_conditioning_node.py
        cond_path = os.path.join(REPO_ROOT, "nodes", "h3_conditioning_node.py")
        with open(cond_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("audio_slot_to_ordinal", src)
        self.assertIn("not has_any_vid_audio", src)

        # Simulate the exact remap rule from h3_conditioning_node.py
        audio_slot_map = {2: 1}
        raw_prompt = "<Audio 2> 愤怒低吼，同时 <Audio 2> 叹息，而 <Audio 3> 不变"

        def _apply_remap(prompt_txt: str, slot_map: dict, audio_drive_active: bool, has_any_vid_audio: bool) -> str:
            if (not audio_drive_active) and (not has_any_vid_audio) and isinstance(slot_map, dict) and slot_map:
                def _remap_audio_tag(m):
                    slot_k = int(m.group(1))
                    mapped = slot_map.get(slot_k)
                    return f"<Audio {mapped}>" if mapped else m.group(0)
                return re.sub(r"<Audio\s+(\d+)>", _remap_audio_tag, prompt_txt)
            return prompt_txt

        remapped = _apply_remap(raw_prompt, audio_slot_map, False, False)
        self.assertEqual(remapped, "<Audio 1> 愤怒低吼，同时 <Audio 1> 叹息，而 <Audio 3> 不变")

        # When a soundtracked ref video exists, DO NOT remap <Audio k>
        untouched_vid = _apply_remap(raw_prompt, audio_slot_map, False, True)
        self.assertEqual(untouched_vid, raw_prompt)

        # When audio_drive_active is True, DO NOT remap
        untouched_drv = _apply_remap(raw_prompt, audio_slot_map, True, False)
        self.assertEqual(untouched_drv, raw_prompt)

    def test_t4_audio_drive_strict_mute(self):
        shot = {
            "shot_index": 1,
            "inherit_global_refs": True,
            "inherit_global_audios": True,
            "ref_slots": [""] * 9,
            "ref_audio_slots": ["own1.wav", "v2.wav", ""],
            "ref_audios": ["own1.wav", "v2.wav"],
        }
        gcfg = {"global_ref_audio_slots": ["g1.wav", "g2.wav", "v3.wav"]}
        with patch.object(aat, "_load_audio_dict", side_effect=_fake_audio_loader) as mocked_load:
            mm = aat.load_shot_multimodal_refs(
                shot,
                global_cfg=gcfg,
                input_dir=self.root,
                skip_ref_audios=True,
            )
            self.assertEqual(mm["audios"], [])
            self.assertEqual(mm["audio_slot_to_ordinal"], {})
            self.assertEqual(mm["slot_audios"], [None, None, None])
            mocked_load.assert_not_called()

    def test_t5_is_changed_invalidates_on_voice_edit(self):
        base_pipe = {
            "project_name": "P15CacheTest",
            "project_root": self.root,
            "fps": 24,
            "width": 1280,
            "height": 720,
            "context_frames": 22,
            "audio_drive_active": False,
            "global_refs_cfg": {
                "global_ref_slots": [""] * 9,
                "global_ref_images": [],
                "global_ref_audio_slots": ["g1.wav", "", ""],
                "global_ref_audios": ["g1.wav"],
            },
            "retry_lock": {"enabled": False},
            "shots": [
                {
                    "shot_index": 1,
                    "prompt": "test shot",
                    "prompt_sha256": "abc123",
                    "duration_sec": 5.0,
                    "shot_type": 1,
                    "hard_cut": True,
                    "overlap_override": 0,
                    "seed_text": "",
                    "inherit_global_refs": True,
                    "inherit_global_audios": True,
                    "ref_slots": [""] * 9,
                    "ref_images": [],
                    "ref_videos": [],
                    "ref_audio_slots": ["", "", ""],
                    "ref_audios": [],
                    "guide_audio": "",
                }
            ],
        }

        h_disp_base = ShotVaultLoopDispatch.IS_CHANGED(shot_pipe=base_pipe, segment_index=0)
        h_load_base = ShotVaultLoad.IS_CHANGED(
            shot_pipe=base_pipe,
            start_mode="manual_picks",
            shot_numbers="1",
            continue_from="auto_latent_first",
            seed_mode="fixed",
        )

        # 1. Change shot ref_audio_slots
        p_shot_aud = json.loads(json.dumps(base_pipe))
        p_shot_aud["shots"][0]["ref_audio_slots"] = ["", "v2.wav", ""]
        p_shot_aud["shots"][0]["ref_audios"] = ["v2.wav"]
        self.assertNotEqual(h_disp_base, ShotVaultLoopDispatch.IS_CHANGED(shot_pipe=p_shot_aud, segment_index=0))
        self.assertNotEqual(
            h_load_base,
            ShotVaultLoad.IS_CHANGED(
                shot_pipe=p_shot_aud,
                start_mode="manual_picks",
                shot_numbers="1",
                continue_from="auto_latent_first",
                seed_mode="fixed",
            ),
        )

        # 2. Change global_ref_audio_slots
        p_glob_aud = json.loads(json.dumps(base_pipe))
        p_glob_aud["global_refs_cfg"]["global_ref_audio_slots"] = ["g2.wav", "", ""]
        p_glob_aud["global_refs_cfg"]["global_ref_audios"] = ["g2.wav"]
        self.assertNotEqual(h_disp_base, ShotVaultLoopDispatch.IS_CHANGED(shot_pipe=p_glob_aud, segment_index=0))
        self.assertNotEqual(
            h_load_base,
            ShotVaultLoad.IS_CHANGED(
                shot_pipe=p_glob_aud,
                start_mode="manual_picks",
                shot_numbers="1",
                continue_from="auto_latent_first",
                seed_mode="fixed",
            ),
        )

        # 3. Toggle inherit_global_audios
        p_toggle = json.loads(json.dumps(base_pipe))
        p_toggle["shots"][0]["inherit_global_audios"] = False
        self.assertNotEqual(h_disp_base, ShotVaultLoopDispatch.IS_CHANGED(shot_pipe=p_toggle, segment_index=0))
        self.assertNotEqual(
            h_load_base,
            ShotVaultLoad.IS_CHANGED(
                shot_pipe=p_toggle,
                start_mode="manual_picks",
                shot_numbers="1",
                continue_from="auto_latent_first",
                seed_mode="fixed",
            ),
        )

    def test_t6_frontend_voice_reuse_ui_and_zero_chinese_leak(self):
        panel_path = os.path.join(REPO_ROOT, "web", "director_panel.js")
        i18n_path = os.path.join(REPO_ROOT, "web", "i18n.js")
        styles_path = os.path.join(REPO_ROOT, "web", "panel", "styles.js")

        with open(panel_path, "r", encoding="utf-8") as f:
            panel_js = f.read()
        with open(i18n_path, "r", encoding="utf-8") as f:
            i18n_js = f.read()
        with open(styles_path, "r", encoding="utf-8") as f:
            styles_js = f.read()

        # Verify zero raw Chinese characters leaked into director_panel.js
        leaks = []
        for idx, line in enumerate(panel_js.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            # Allow existing regex character patterns like /文戏|武戏/
            if "mapShotTypeDirective" in line or "splitMasterScript" in line or "match(" in line or ".test(" in line or ".replace(" in line:
                continue
            if CJK_RE.search(line):
                leaks.append((idx, line.strip()[:120]))
        self.assertEqual(leaks, [], f"Hardcoded Chinese found in director_panel.js: {leaks}")

        # Verify all required v5.8.4 helpers & buttons are wired
        for token in [
            "function normSlots3(",
            "function resolveEffectiveAudioSlots(",
            "function render3AudioSlotGrid(",
            "function stopAudioPreview(",
            't("btn_copy_voices_forward")',
            't("btn_apply_voices_all")',
            't("btn_copy_global_voices")',
            't("lbl_inherit_global_audios")',
            't("lbl_shot_audio_slots"',
            't("warn_video_audio_shift"',
        ]:
            self.assertIn(token, panel_js, f"Missing expected frontend symbol: {token}")

        # Option B check: btn_apply_voices_all must NOT call window.confirm
        apply_idx = panel_js.index('t("btn_apply_voices_all")')
        apply_slice = panel_js[apply_idx : apply_idx + 900]
        self.assertNotIn("confirm(", apply_slice)
        self.assertIn("state.plan.global_ref_audio_slots = [...resolved]", apply_slice)
        self.assertIn("sh.inherit_global_audios = true", apply_slice)

        # Verify CSS classes exist
        for cls in [".sv-aud-box", ".sv-aud-grid3", ".sv-aud-slot.own", ".sv-aud-slot.inh", ".sv-aud-slot.muted", ".sv-aud-warn"]:
            self.assertIn(cls, styles_js)

        # Verify ZH/EN key symmetry for new v5.8.4 keys
        for key in [
            "lbl_shot_audio_slots",
            "lbl_inherit_global_audios",
            "btn_copy_voices_forward",
            "btn_apply_voices_all",
            "toast_voices_copied_forward",
            "toast_voices_applied_all",
            "toast_no_voices_to_copy",
            "aud_slot_own",
            "aud_slot_inherited",
            "aud_slot_empty",
            "aud_slot_upload_title",
            "btn_play_audio",
            "btn_stop_audio",
            "title_play_audio",
            "warn_video_audio_shift",
            "btn_copy_global_voices",
            "toast_global_voices_copied",
            "lbl_global_voice_hdr",
        ]:
            self.assertEqual(len(re.findall(rf"^\s*{key}\s*:", i18n_js, re.M)), 2, f"Key {key} must appear in both zh and en")


if __name__ == "__main__":
    unittest.main()
