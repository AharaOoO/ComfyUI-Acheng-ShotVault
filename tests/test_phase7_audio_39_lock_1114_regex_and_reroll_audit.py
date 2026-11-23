"""Phase 7 tests: Audio-drive mandatory 39f lock, [overlap+curr] real audio guide, 1114 &duration& split, reroll lock, Shot0 & sparse 9-ref ports."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import wave
from unittest.mock import patch
import unittest

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
parser_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.script_parser")
color_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.seam_and_color")


def _write_wav(path: str, amp: float, dur_sec: float = 3.0, sr: int = 48000):
    n = round(dur_sec * sr)
    t = np.arange(n, dtype=np.float32) / float(sr)
    sig = (amp * np.sin(2.0 * np.pi * 440.0 * t)).clip(-0.99, 0.99)
    pcm = (np.stack([sig, sig], axis=-1) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class _CaptureClip:
    def __init__(self):
        self.prompts = []

    def tokenize(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {"prompt": prompt, "refs": kwargs.get("minimax_ref_items") or []}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones((1, 4, 8), dtype=torch.float32), {"captured": tokens["prompt"]}]]


class TestPhase7Audio39Lock1114RegexAndRerollAudit(unittest.TestCase):
    def test_audio_drive_mode_forces_immutable_39f_and_regular_allows_22_or_39(self):
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        disp_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoopDispatch"]()

        with tempfile.TemporaryDirectory() as tmp:
            aud_dir = os.path.join(tmp, "aud")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(aud_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            _write_wav(os.path.join(aud_dir, "01.wav"), 0.5, 3.0)
            _write_wav(os.path.join(aud_dir, "02.wav"), 0.3, 3.0)

            bad_json = json.dumps({
                "audio_drive_mode": "per_shot_audio_dir",
                "context_frames": 22,
                "shots": [
                    {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "hard_cut": True, "overlap_override": 22, "prompt": "第一段"},
                    {"shot_index": 2, "duration_sec": 3.0, "shot_type": 1, "hard_cut": False, "overlap_override": 22, "prompt": "第二段"},
                ],
            })
            with patch("folder_paths.get_output_directory", return_value=out_dir):
                pipe_aud = plan_cls.plan("AudioLock", "auto", "c", "", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", audio_drive_mode="per_shot_audio_dir", audio_per_shot_dir=aud_dir, director_plan_json=bad_json)[0]
                self.assertTrue(pipe_aud["audio_drive_active"])
                self.assertEqual(pipe_aud["context_frames"], 39)

                # Save shot 1 so shot 2 resolves real continuation
                p1, *_ = load_cls.load_shot(pipe_aud, "manual_picks", "1", "pixel_clean_taper", "fixed", 10, "", 0.0)
                sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]().save_take(torch.rand((p1["keep_frames"], 64, 64, 3)), p1, False, 20, "h264")

                p2, *_ = load_cls.load_shot(pipe_aud, "manual_picks", "2", "pixel_clean_taper", "fixed", 10, "", 0.0)
                self.assertEqual(p2["overlap_frames"], 39)
                self.assertEqual(p2["trim_head_frames"], 39)
                self.assertEqual(p2["gear_feather_ticks"], 12)

                d2 = disp_cls.dispatch(pipe_aud, segment_index=1, index_base="0_based (easy_forLoop)", honor_director_retry_lock=False)
                self.assertEqual(d2[10], 39)
                self.assertEqual(d2[11], 39)
                self.assertEqual(d2[15], 12)

                # Now regular mode ("none") freely allows 22f vs 39f!
                reg_json = json.dumps({
                    "audio_drive_mode": "none",
                    "context_frames": 22,
                    "shots": [
                        {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "hard_cut": True, "prompt": "常规1"},
                        {"shot_index": 2, "duration_sec": 3.0, "shot_type": 1, "hard_cut": False, "prompt": "常规2"},
                    ],
                })
                pipe_reg = plan_cls.plan("RegFree", "auto", "c", "", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", audio_drive_mode="none", director_plan_json=reg_json)[0]
                self.assertFalse(pipe_reg["audio_drive_active"])
                self.assertEqual(pipe_reg["context_frames"], 22)
                d2_reg = disp_cls.dispatch(pipe_reg, segment_index=1, index_base="0_based (easy_forLoop)", honor_director_retry_lock=False)
                self.assertEqual(d2_reg[10], 22)
                self.assertEqual(d2_reg[15], 7)

    def test_per_shot_and_master_song_overlap_audio_concat_zero_silence_cliff(self):
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        save_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()

        with tempfile.TemporaryDirectory() as tmp:
            aud_dir = os.path.join(tmp, "aud")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(aud_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            _write_wav(os.path.join(aud_dir, "01.wav"), 0.65, 3.0)
            _write_wav(os.path.join(aud_dir, "02.wav"), 0.20, 3.0)

            with patch("folder_paths.get_output_directory", return_value=out_dir):
                pipe = plan_cls.plan("AudConcat", "auto", "c", "& 3.0 &\nA\n===\n& 3.0 &\nB", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", audio_drive_mode="per_shot_audio_dir", audio_per_shot_dir=aud_dir)[0]
                p1, *_ = load_cls.load_shot(pipe, "manual_picks", "1", "pixel_clean_taper", "fixed", 10, "", 0.0)
                save_cls.save_take(torch.rand((p1["keep_frames"], 32, 32, 3)), p1, False, 20, "h264")
                p2, *_ = load_cls.load_shot(pipe, "manual_picks", "2", "pixel_clean_taper", "fixed", 10, "", 0.0)
                guide = p2["shot_audio_guide"]["waveform"][0]
                pre_samples = round(39 / 24.0 * 32000)  # Exact 52,000 samples (65 AudioVAE tokens)
                self.assertEqual(pre_samples, 52000)
                self.assertGreater(float(guide[:, :pre_samples].abs().max().item()), 0.50)
                self.assertAlmostEqual(float(guide[:, pre_samples:pre_samples + 8000].abs().max().item()), 0.20, delta=0.05)

                # Verify delivery audio stays strictly net shot 2 (3.0s) and trim_av_sync leaves 0 missing frames!
                deliv = p2["delivery_ref_audio"]["waveform"]
                self.assertAlmostEqual(deliv.shape[-1] / float(p2["delivery_ref_audio"]["sample_rate"]), 3.0, places=2)
                img_t, aud_t = color_mod.trim_av_sync(torch.rand((p2["target_frames"], 32, 32, 3)), p2["delivery_ref_audio"], trim_head_frames=p2["trim_head_frames"], keep_frames=p2["keep_frames"], fps=24)
                self.assertEqual(int(img_t.shape[0]), 72)
                self.assertEqual(int(aud_t["waveform"].shape[-1]), round(72 / 24.0 * 48000))

                # Master song slice mode also populates non-zero 39f overlap tail before Shot 2 start!
                sr_m = 48000
                t = torch.arange(round(7.0 * sr_m), dtype=torch.float32) / sr_m
                song = (0.6 * torch.sin(2 * np.pi * 330 * t)).unsqueeze(0).repeat(2, 1).unsqueeze(0)
                pipe_m = plan_cls.plan("SongSlice", "auto", "c", "& 3.0 &\nA\n===\n& 3.0 &\nB", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", audio_drive_mode="master_song_slice", master_audio_in={"waveform": song, "sample_rate": sr_m})[0]
                pm1, *_ = load_cls.load_shot(pipe_m, "manual_picks", "1", "pixel_clean_taper", "fixed", 10, "", 0.0)
                save_cls.save_take(torch.rand((pm1["keep_frames"], 32, 32, 3)), pm1, False, 20, "h264")
                pm2, *_ = load_cls.load_shot(pipe_m, "manual_picks", "2", "pixel_clean_taper", "fixed", 10, "", 0.0)
                gm = pm2["shot_audio_guide"]["waveform"][0]
                self.assertGreater(float(gm[:, :pre_samples].abs().max().item()), 0.45)

    def test_1114_regex_ampersand_duration_inline_delim_and_default_duration(self):
        script = "& 8 &\n第一段武戏起手===& 6.5 &\n第二段连续交锋===\n第三段省略时长走默认值==="
        plan_json = json.dumps({"master_script_text": script, "default_duration_sec": 7.0})
        gcfg, shots = parser_mod.parse_director_plan_json(plan_json, common_prompt="电影画风", fps=24)
        self.assertEqual(gcfg["default_duration_sec"], 7.0)
        self.assertEqual(len(shots), 3)
        self.assertAlmostEqual(shots[0]["duration_sec"], 8.0, places=2)
        self.assertAlmostEqual(shots[1]["duration_sec"], 6.5, places=2)
        self.assertAlmostEqual(shots[2]["duration_sec"], 7.0, places=2)
        self.assertEqual(shots[0]["local_prompt"], "第一段武戏起手")
        self.assertEqual(shots[1]["local_prompt"], "第二段连续交锋")
        self.assertEqual(shots[2]["local_prompt"], "第三段省略时长走默认值")

        js_path = os.path.join(PKG_DIR, "web", "director_panel.js")
        proc = subprocess.run(["node", "--check", js_path], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
        for tok in ("parseSingleShotAmpersand", "汇总为&时长&+===", "⏱️默认时长(s):", "🔒 39帧 (音频驱动固定)", "🎬 常规模式", "🎵 音频驱动"):
            self.assertIn(tok, js)

    def test_shotvault_load_and_dispatch_retry_lock_fixed_and_delta_plus_1(self):
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        disp_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoopDispatch"]()
        save_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()

        with tempfile.TemporaryDirectory() as tmp, patch("folder_paths.get_output_directory", return_value=tmp):
            base_plan = json.dumps({
                "shots": [
                    {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "hard_cut": True, "prompt": "首段"},
                    {"shot_index": 2, "duration_sec": 3.0, "shot_type": 0, "hard_cut": False, "prompt": "二段初版"},
                ],
            })
            pipe0 = plan_cls.plan("RerollLock", "auto", "高清电影", "", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", director_plan_json=base_plan)[0]
            p2_init, *_ = load_cls.load_shot(pipe0, "manual_picks", "2", "pixel_clean_taper", "fixed", 5000, "", 0.0)
            save_cls.save_take(torch.rand((p2_init["keep_frames"], 64, 64, 3)), p2_init, False, 20, "h264")

            delta_plan = json.dumps({
                "retry_lock": {"enabled": True, "shot_index": 2, "seed_mode": "delta_+1", "seed_text": "", "prompt_override": "二段精修特写"},
                "shots": [
                    {"shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "hard_cut": True, "prompt": "首段"},
                    {"shot_index": 2, "duration_sec": 3.0, "shot_type": 0, "hard_cut": False, "prompt": "二段初版"},
                ],
            })
            pipe_delta = plan_cls.plan("RerollLock", "auto", "高清电影", "", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", director_plan_json=delta_plan)[0]
            p_retry = load_cls.load_shot(pipe_delta, "sequence_auto", "1", "pixel_clean_taper", "fixed", 100, "", 0.0)
            self.assertEqual(p_retry[0]["current_shot_index"], 2)
            self.assertEqual(p_retry[0]["queue_remaining"], [])
            self.assertIn("二段精修特写", p_retry[3])
            self.assertEqual(p_retry[7], 5001)

            d_retry = disp_cls.dispatch(pipe_delta, segment_index=0, index_base="0_based (easy_forLoop)", honor_director_retry_lock=True)
            self.assertEqual(d_retry[2], 2)
            self.assertEqual(d_retry[16], 5001)

    def test_sparse_9_ref_slots_preserve_port_positions_and_picture_tags(self):
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        disp_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoopDispatch"]()
        cond_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultH3Conditioning"]()

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(os.path.join(in_dir, "slot2_hero.png"))

            sparse_plan = json.dumps({
                "global_ref_slots": [""] * 9,
                "shots": [
                    {
                        "shot_index": 1, "duration_sec": 3.0, "shot_type": 1, "hard_cut": True,
                        "prompt": "<Picture 2> 转身长枪横扫",
                        "ref_slots": ["", "slot2_hero.png", "", "", "", "", "", "", ""],
                    },
                ],
            })
            with patch("folder_paths.get_input_directory", return_value=in_dir), patch("folder_paths.get_output_directory", return_value=out_dir):
                pipe = plan_cls.plan("SparseRefs", "auto", "", "", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", director_plan_json=sparse_plan)[0]
                res = disp_cls.dispatch(pipe, segment_index=0, index_base="0_based (easy_forLoop)")
                self.assertIsNone(res[17])        # ref_image_1 is None!
                self.assertIsNotNone(res[18])     # ref_image_2 holds Slot 2 tensor!
                clip = _CaptureClip()
                cond_cls.encode_shot(clip, res[0])
                self.assertIn("<Picture 1> 转身长枪横扫", clip.prompts[0])

    def test_loop_dispatch_loads_shot0_seed_context(self):
        plan_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        disp_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoopDispatch"]()
        qc_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.seam_qc")

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            seed_mp4 = os.path.join(in_dir, "prequel.mp4")
            aud_48k = torch.sin(torch.linspace(0, 80, round(40 / 24.0 * 48000))).repeat(2, 1) * 0.4
            qc_mod.write_mp4_with_pcm(torch.rand((40, 64, 64, 3)), aud_48k, 48000, 24, seed_mp4)

            with patch("folder_paths.get_input_directory", return_value=in_dir), patch("folder_paths.get_output_directory", return_value=out_dir):
                asset_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.asset_and_turnaround")
                ref_loaded = asset_mod._load_video_frames_and_audio(seed_mp4, target_fps=24, max_frames=40)
                self.assertIsNotNone(ref_loaded["audio"])
                self.assertGreater(float(ref_loaded["audio"]["waveform"].abs().max().item()), 0.05)

                pipe = plan_cls.plan("Shot0Disp", "auto", "c", "& 3.0 &\n续拍起手", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22", seed_video_path="prequel.mp4")[0]
                res = disp_cls.dispatch(pipe, segment_index=0, index_base="0_based (easy_forLoop)")
                self.assertTrue(res[9])  # is_continuation=True
                self.assertEqual(res[0]["parent_take_id"], "shot_0000_seed")
                self.assertGreater(float(res[0]["prev_audio"]["waveform"].abs().max().item()), 0.05)


if __name__ == "__main__":
    unittest.main()
