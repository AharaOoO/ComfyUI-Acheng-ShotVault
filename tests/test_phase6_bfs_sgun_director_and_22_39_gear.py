"""Phase 6 tests: v5.1 BFSxSGUN Director UI contracts, complete 22/39 gear-shift, LoopDispatch, and seam diagnostics."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image  # type: ignore

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_NODES_DIR = os.path.dirname(PKG_DIR)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES_DIR)
for p in (COMFY_ROOT, CUSTOM_NODES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

sv_pkg = importlib.import_module("ComfyUI-Universal-ShotVault")
grid_math = importlib.import_module("ComfyUI-Universal-ShotVault.core.grid_math")
parser_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.script_parser")
asset_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.asset_and_turnaround")
qc_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.seam_qc")
vs_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.vault_store")
srv_mod = importlib.import_module("ComfyUI-Universal-ShotVault.server_routes")


class _MockAudioVAE:
    audio_sample_rate = 32000

    def encode(self, wav):
        t_40 = max(1, round(int(wav.shape[1]) / 32000.0 * 40.0))
        return torch.ones((1, 32, 2, t_40), dtype=torch.float32) * 0.3


class _MockVideoVAE:
    def encode(self, frames):
        f = int(frames.shape[0])
        t = 2 if f <= 5 else ((f - 5) // 17) * 5 + 2
        h, w = max(1, int(frames.shape[1]) // 16), max(1, int(frames.shape[2]) // 16)
        return torch.ones((1, 24, t, h, w), dtype=torch.float32) * 0.5


class TestPhase6BfsSgunDirectorAnd22_39Gear(unittest.TestCase):
    def test_complete_22_vs_39_vs_56_gear_linkage_and_shot1_taper_safety(self):
        taper_mod = importlib.import_module("ComfyUI-H3-Context-Noise.nodes")
        g22 = grid_math.compute_overlap_gear(22, base_gear=22, is_continuation=True)
        g39 = grid_math.compute_overlap_gear(39, base_gear=39, is_continuation=True)
        g56 = grid_math.compute_overlap_gear(56, base_gear=56, is_continuation=True)
        g1_noncont = grid_math.compute_overlap_gear(0, base_gear=39, is_continuation=False)

        self.assertEqual((g22["overlap_frames"], g22["trim_head_frames"], g22["taper_tail_frames"], g22["taper_ramp_frames"], g22["audio_guide_frames"], g22["audio_feather_ticks"], g22["continuation_pair"]), (22, 22, 22, 4, 20, 7, "22+5"))
        self.assertEqual((g39["overlap_frames"], g39["trim_head_frames"], g39["taper_tail_frames"], g39["taper_ramp_frames"], g39["audio_guide_frames"], g39["audio_feather_ticks"], g39["continuation_pair"]), (39, 39, 39, 6, 37, 12, "39+5"))
        self.assertEqual((g56["overlap_frames"], g56["trim_head_frames"], g56["taper_tail_frames"], g56["taper_ramp_frames"], g56["audio_guide_frames"], g56["audio_feather_ticks"], g56["continuation_pair"]), (56, 56, 56, 8, 54, 16, "56+5"))
        self.assertEqual(g1_noncont["overlap_frames"], 0)
        self.assertGreaterEqual(g1_noncont["taper_tail_frames"], 22)

        sched = grid_math.compute_cumulative_schedule([4.0, 5.0, 4.0], fps=24, is_continuations=[False, True, True], context_frames=22, overlap_overrides=[0, 39, 0])
        self.assertEqual(sched[1]["overlap_frames"], 39)
        self.assertEqual(sched[1]["trim_head_frames"], 39)
        self.assertEqual(sched[2]["overlap_frames"], 22)

        taper = taper_mod.MiniMaxH3ContextTaperNoise()
        dummy_frames = torch.ones((4, 64, 64, 3), dtype=torch.float32) * 0.5
        out_img, _ = taper.inject(dummy_frames, g1_noncont["taper_tail_frames"], 0.55, 0.0, g1_noncont["taper_ramp_frames"], 0)
        self.assertEqual(tuple(out_img.shape), (4, 64, 64, 3))

    def test_master_script_auto_split_and_directives(self):
        master = "@武戏\n@22\n主角起跳劈砍\n---\n@文戏\n@overlap=39\n(6.5s)\n近景呼吸停顿，对视良久\n===\n@56\n终结技慢动作收刀"
        plan_json = json.dumps({"master_script_text": master, "context_frames": 22})
        gcfg, shots = parser_mod.parse_director_plan_json(plan_json, common_prompt="写实电影", fps=24)
        self.assertEqual(gcfg["context_frames"], 22)
        self.assertEqual(len(shots), 3)
        self.assertEqual(shots[0]["shot_type"], 1)
        self.assertEqual(shots[0]["overlap_override"], 22)
        self.assertEqual(shots[1]["shot_type"], 0)
        self.assertEqual(shots[1]["overlap_override"], 39)
        self.assertAlmostEqual(shots[1]["duration_sec"], 6.5, places=2)
        self.assertEqual(shots[2]["overlap_override"], 56)

    def test_9_slot_ref_inheritance_and_copy_to_subsequent(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "input")
            os.makedirs(in_dir, exist_ok=True)
            for fn in ("g1.png", "g2.png", "s2.png"):
                Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(os.path.join(in_dir, fn))

            g_slots = ["g1.png", "g2.png"] + [""] * 7
            s1_slots = ["", ""] + [""] * 7
            s2_slots = ["s2.png", ""] + [""] * 7
            s3_slots = list(s2_slots)  # Simulate one-click copy-to-subsequent from Shot 2 -> Shot 3
            gcfg = {"global_ref_slots": g_slots, "global_ref_images": ["g1.png", "g2.png"]}

            m1 = asset_mod.load_shot_multimodal_refs({"shot_index": 1, "ref_slots": s1_slots, "inherit_global_refs": True}, gcfg, in_dir, tmp)
            m2 = asset_mod.load_shot_multimodal_refs({"shot_index": 2, "ref_slots": s2_slots, "inherit_global_refs": True}, gcfg, in_dir, tmp)
            m3 = asset_mod.load_shot_multimodal_refs({"shot_index": 3, "ref_slots": s3_slots, "inherit_global_refs": True}, gcfg, in_dir, tmp)
            self.assertEqual(len(m1["images"]), 2)
            self.assertEqual(len(m2["images"]), 2)
            self.assertEqual(len(m3["images"]), 2)

    def test_loop_dispatch_0_and_1_based_retry_lock_and_9_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "input")
            out_dir = os.path.join(tmp, "output")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8)).save(os.path.join(in_dir, "hero.png"))

            plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
            disp_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoopDispatch"]()
            plan_json = json.dumps({
                "context_frames": 39,
                "global_ref_slots": ["hero.png"] + [""] * 8,
                "retry_lock": {"enabled": True, "shot_index": 2, "seed_mode": "custom", "seed_text": "778899", "prompt_override": "单段重抽微调特写"},
                "shots": [
                    {"shot_index": 1, "duration_sec": 4.0, "shot_type": 1, "hard_cut": True, "prompt": "第一段"},
                    {"shot_index": 2, "duration_sec": 5.0, "shot_type": 0, "hard_cut": False, "prompt": "第二段原词"},
                ],
            })
            with patch("folder_paths.get_output_directory", return_value=out_dir), patch("folder_paths.get_input_directory", return_value=in_dir):
                pipe = plan_cls.plan("LoopTest", "auto", "", "", "16:9 (1344x768)", 1.0, 24, "cumulative_cursor", "39", director_plan_json=plan_json)[0]
                res_lock = disp_cls.dispatch(pipe, segment_index=0, index_base="0_based (easy_forLoop)", honor_director_retry_lock=True)
                self.assertEqual(len(res_lock), 27)
                self.assertEqual(res_lock[1], 2)       # total_segments
                self.assertEqual(res_lock[2], 2)       # current_segment locked to shot 2
                self.assertIn("单段重抽微调特写", res_lock[3])
                self.assertTrue(res_lock[8])           # is_drama_mode=True
                self.assertEqual(res_lock[10], 39)     # overlap_frames=39
                self.assertEqual(res_lock[11], 39)     # trim_head_frames=39
                self.assertEqual(res_lock[12], 39)     # taper_tail_frames=39
                self.assertEqual(res_lock[14], 37)     # audio_guide_frames=37
                self.assertEqual(res_lock[15], 12)     # audio_feather_ticks=12
                self.assertEqual(res_lock[16], 778899) # seed overridden cleanly
                self.assertEqual(tuple(res_lock[17].shape[1:]), (64, 64, 3))
                self.assertEqual(res_lock[26].ndim, 4)  # prev_clean_images slot [26]

                res_1based = disp_cls.dispatch(pipe, segment_index=1, index_base="1_based (1..N)", honor_director_retry_lock=False)
                self.assertEqual(res_1based[2], 1)
                self.assertEqual(res_1based[10], 0)    # hard_cut shot 1 -> overlap 0
                self.assertGreaterEqual(res_1based[12], 39)  # #749 safety on shot 1

    def test_window_guider_auto_plus_5_gear_shifting_22_vs_39(self):
        import comfy.nested_tensor  # type: ignore

        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        lat = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 17, 2, 2)), torch.zeros((1, 32, 2, 93))))}
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        prev_40 = torch.ones((40, 32, 32, 3), dtype=torch.float32) * 0.4
        guide_audio = {"waveform": torch.zeros((1, 2, 48000)), "sample_rate": 48000}

        pipe22 = {"is_continuation": True, "overlap_frames": 22, "gear_feather_ticks": 7, "auto_scale_feather": True}
        pipe39 = {"is_continuation": True, "overlap_frames": 39, "gear_feather_ticks": 12, "auto_scale_feather": True}

        p22, _ = wg.prepare_guider(None, pos, lat, "single_or_1st", "off", 192, 39, "auto+5", 0.85, 7, True, False, 1.0, vae=_MockVideoVAE(), audio_vae=_MockAudioVAE(), tapered_prev_images=prev_40, shot_audio_guide=guide_audio, shot_pipe=pipe22)
        p39, _ = wg.prepare_guider(None, pos, lat, "single_or_1st", "off", 192, 39, "auto+5", 0.85, 7, True, False, 1.0, vae=_MockVideoVAE(), audio_vae=_MockAudioVAE(), tapered_prev_images=prev_40, shot_audio_guide=guide_audio, shot_pipe=pipe39)
        p39_2nd, _ = wg.prepare_guider(None, pos, lat, "2nd_refine", "off", 192, 39, "auto+5", 0.85, 7, True, False, 1.0, vae=_MockVideoVAE(), audio_vae=_MockAudioVAE(), tapered_prev_images=prev_40, shot_pipe=pipe39)

        vkf22 = [k["latent"] for k in p22[0][1]["minimax_keyframes"] if k.get("latent") is not None][0]
        vkf39 = [k["latent"] for k in p39[0][1]["minimax_keyframes"] if k.get("latent") is not None][0]
        vkf_2nd = [k["latent"] for k in p39_2nd[0][1]["minimax_keyframes"] if k.get("latent") is not None][0]
        self.assertEqual(vkf22.shape[2], 7)   # 22f -> 7 latent tokens
        self.assertEqual(vkf39.shape[2], 12)  # 39f -> 12 latent tokens
        self.assertEqual(vkf_2nd.shape[2], 2) # auto+5 -> 5f on 2nd_refine -> 2 latent tokens

    def test_seam_diagnostics_jump_detection_and_vault_save(self):
        smooth_prev = torch.full((24, 64, 64, 3), 0.35, dtype=torch.float32)
        smooth_curr = torch.full((48, 64, 64, 3), 0.36, dtype=torch.float32)
        ok_diag = qc_mod.analyze_seam_diagnostics(smooth_prev, smooth_curr, fps=24)
        self.assertEqual(ok_diag["status"], "ok")
        self.assertEqual(ok_diag["join_frame"], 24)

        jump_curr = torch.full((48, 64, 64, 3), 0.85, dtype=torch.float32)
        warn_diag = qc_mod.analyze_seam_diagnostics(smooth_prev, jump_curr, fps=24)
        self.assertEqual(warn_diag["status"], "warning")
        self.assertGreaterEqual(len(warn_diag["events"]), 1)

        with tempfile.TemporaryDirectory() as tmp, patch("folder_paths.get_output_directory", return_value=tmp):
            root = srv_mod.resolve_output_root("SeamProj")
            store = vs_mod.VaultStore(root)
            meta = store.save_take_context(
                take_id="s2_t001", shot_index=2, clean_images=jump_curr,
                audio_waveform_48k=torch.zeros((2, 96000)), audio_sr=48000, fps=24,
                prompt_sha256="abc", parent_take_id="s1_t1", seed_text="11",
                keep_frames=48, trim_head_frames=22, shot_type=1,
                parent_clean_tail=smooth_prev, seam_diagnostics=warn_diag,
            )
            self.assertEqual(meta["seam_diagnostics"]["status"], "warning")
            st = srv_mod._build_project_state_payload("SeamProj")
            self.assertEqual(st["shots"][0]["takes"][0]["seam_diagnostics"]["status"], "warning")

    def test_director_panel_v51_bfs_sgun_contracts_and_zero_internal_loop(self):
        js_path = os.path.join(PKG_DIR, "web", "director_panel.js")
        proc = subprocess.run(["node", "--check", js_path], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
        i18n_path = os.path.join(PKG_DIR, "web", "i18n.js")
        if os.path.isfile(i18n_path):
            with open(i18n_path, "r", encoding="utf-8") as f:
                js += "\n" + f.read()
        for req in ('w.type = "hidden"', "w.hidden = true", "由连线控制", "splitMasterScript", "render9SlotGrid", "一键沿用本段9图到后续所有段", "单段重抽锁定", "22 / 39 / 56 帧完整换挡联动矩阵", "reloadFromWidgets"):
            self.assertIn(req, js)
        self.assertNotIn("api.queuePrompt", js)

    def test_v51_workflows_include_loop_dispatch_dynamic_749_and_auto_plus_5(self):
        wf_dir = os.path.join(PKG_DIR, "example_workflows")
        wf_files = sorted(fn for fn in os.listdir(wf_dir) if fn.endswith(".json"))
        self.assertEqual(len(wf_files), 5)
        for fn in wf_files:
            with open(os.path.join(wf_dir, fn), "r", encoding="utf-8") as f:
                wf = json.load(f)
            by_type = {}
            for n in wf["nodes"]:
                by_type.setdefault(n["type"], []).append(n)
            self.assertEqual(len(by_type.get("ShotVaultLoopDispatch", [])), 1, fn)
            taper = by_type["MiniMaxH3ContextTaperNoise"][0]
            in_names = [x["name"] for x in taper.get("inputs", [])]
            self.assertIn("tail_frames", in_names)
            self.assertIn("ramp_frames", in_names)
            for wg in by_type["ShotVaultWindowGuider"]:
                self.assertEqual(wg["widgets_values"][4], "auto+5", fn)


if __name__ == "__main__":
    unittest.main()
