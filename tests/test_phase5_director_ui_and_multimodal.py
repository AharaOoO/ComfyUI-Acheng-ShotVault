"""Phase 5 tests: v5.0 Visual Director Console UI, ShotVaultH3Conditioning, Shot 0 audio/latent, and dual-hemisphere workflows."""

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

sv_pkg = importlib.import_module("ComfyUI-Acheng-ShotVault")
grid_math = importlib.import_module("ComfyUI-Acheng-ShotVault.core.grid_math")
parser_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.script_parser")
qc_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.seam_qc")
seed_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.seed_video_ingest")
vs_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.vault_store")
srv_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.server_routes")


class _CaptureClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return {"prompt": prompt, "refs": kwargs.get("minimax_ref_items") or []}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones((1, 4, 8), dtype=torch.float32), {"captured_refs": tokens["refs"]}]]


class _MockVAE:
    def encode(self, frames):
        f = int(frames.shape[0])
        h, w = max(1, int(frames.shape[1]) // 16), max(1, int(frames.shape[2]) // 16)
        vt = grid_math.video_latent_t(max(5, f)) if f > 1 else 1
        return torch.ones((1, 24, vt, h, w), dtype=torch.float32) * 0.6


class _MockAudioVAE:
    audio_sample_rate = 32000

    def encode(self, wav):
        # Record encoded sample count to verify 800-sample alignment
        t_40 = max(1, round(int(wav.shape[1]) / 32000.0 * 40.0))
        return torch.ones((1, 32, 2, t_40), dtype=torch.float32) * 0.3


class TestPhase5DirectorUIAndMultimodal(unittest.TestCase):
    def test_director_plan_json_multimodal_parse_and_hard_cut(self):
        cfg = {
            "version": "5.0",
            "seed_video_path": "shot0.mp4",
            "seed_trim_end_sec": 3.5,
            "global_ref_images": ["g1.png"],
            "global_ref_audios": ["g_voice.wav"],
            "shots": [
                {
                    "shot_index": 1, "duration_sec": 4.5, "shot_type": 0, "hard_cut": True,
                    "inherit_global_refs": False, "prompt": "<Picture 1> 人物微笑",
                    "ref_images": ["s1.png"], "ref_videos": [{"path": "act.mp4", "with_audio": True}],
                    "ref_audios": ["s1_voice.wav"], "guide_audio": "beat1.wav",
                },
                {
                    "shot_index": 2, "duration_sec": 6.0, "shot_type": 2, "hard_cut": False,
                    "prompt": "<Picture 1> 长拳", "ref_images": [], "ref_videos": [], "ref_audios": [],
                },
            ],
        }
        gcfg, shots = parser_mod.parse_director_plan_json(json.dumps(cfg), common_prompt="写实电影", fps=24)
        self.assertEqual(gcfg["seed_video_path"], "shot0.mp4")
        self.assertEqual(gcfg["seed_trim_end_sec"], 3.5)
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["shot_type"], 0)
        self.assertTrue(shots[0]["is_drama_mode"])
        self.assertFalse(shots[0]["is_continuation"])
        self.assertFalse(shots[0]["inherit_global_refs"])
        self.assertEqual(shots[0]["audio_file"], "beat1.wav")
        self.assertEqual(shots[1]["shot_type"], 2)
        self.assertTrue(shots[1]["is_continuation"])

    def test_h3_conditioning_real_ref_ordering_and_zero_ref_cleanliness(self):
        cond_node = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultH3Conditioning"]()
        clip = _CaptureClip()
        vae = _MockVAE()
        avae = _MockAudioVAE()

        pipe_with_refs = {
            "prompt": "<Picture 1> <Picture 2> <Audio 1> <Video 1> <Audio 2> 动作戏",
            "width": 1344,
            "height": 768,
            "target_frames": 124,
            "loaded_ref_images": [torch.rand((1, 512, 512, 3)), torch.rand((1, 768, 512, 3))],
            "loaded_ref_videos": [{
                "frames": torch.rand((30, 512, 768, 3)),  # 30f -> snapped down to 22f (17*1+5)
                "audio": {"waveform": torch.ones((1, 2, 32111)), "sample_rate": 32000},
                "name": "action_ref.mp4",
            }],
            "loaded_ref_audios": [{"waveform": torch.ones((1, 2, 16000)), "sample_rate": 32000}],
        }
        pos, lat, ref_map = cond_node.encode_shot(clip, pipe_with_refs, "match", vae=vae, audio_vae=avae)
        ref_items = clip.calls[-1]["kwargs"]["minimax_ref_items"]
        self.assertEqual([r["type"] for r in ref_items], ["image", "image", "audio", "video", "audio"])
        blocks = pos[0][1]["minimax_refs"]
        self.assertEqual(blocks[2]["kind"], "video_audio")
        self.assertEqual(blocks[2]["latent_t"], 7)  # 22 frames == 7 latent tokens
        self.assertIn("<Picture 1>", ref_map)
        self.assertIn("<Video 1>", ref_map)
        self.assertIn("<Audio 2>", ref_map)
        v_lat, _ = lat["samples"].unbind()
        self.assertEqual(tuple(v_lat.shape), (1, 24, grid_math.video_latent_t(124), 48, 84))

        pipe_empty = {
            "prompt": "纯文本生成", "width": 1344, "height": 768, "target_frames": 124,
            "loaded_ref_images": [], "loaded_ref_videos": [], "loaded_ref_audios": [],
        }
        pos0, _, map0 = cond_node.encode_shot(clip, pipe_empty, "match", vae=vae, audio_vae=avae)
        self.assertNotIn("minimax_ref_items", clip.calls[-1]["kwargs"])
        self.assertNotIn("minimax_refs", pos0[0][1])
        self.assertEqual(map0, "")

    def test_load_returns_none_not_black_placeholder_when_no_assets(self):
        plan_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("folder_paths.get_output_directory", return_value=tmp):
                pipe0, _, _ = plan_n.plan("NoAssetProj", "auto", "c", "& 2.0 &\n出拳", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22")
                res = load_n.load_shot(pipe0, "manual_picks", "1", "auto_latent_first", "fixed", 1, "", 0.0)
                self.assertIsNone(res[8])
                self.assertIsNone(res[9])

    def test_shot0_reuse_seed_audio_and_direct_tail_latent_preencode(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = vs_mod.VaultStore(os.path.join(tmp, "SeedProj"))
            frames = torch.rand((48, 64, 64, 3), dtype=torch.float32)
            audio = {"waveform": torch.ones((1, 2, 48000 * 2)) * 0.35, "sample_rate": 48000}
            meta = seed_mod.ingest_seed_video(
                store, width=64, height=64, fps=24,
                reuse_seed_images=frames, reuse_seed_audio=audio,
                vae=_MockVAE(), audio_vae=_MockAudioVAE(),
            )
            self.assertTrue(meta["has_tail_latent"])
            ctx = store.load_take_context("shot_0000_seed")
            self.assertEqual(ctx["tensors"]["tail_latent_video"].shape[2], 12)  # 39 frames == 12 tokens!
            self.assertGreater(float(ctx["tensors"]["delivery_audio_48k"].abs().mean().item()), 0.2)

    def test_per_shot_director_upload_and_multimodal_e2e_plan_to_conditioning(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp_dir = os.path.join(tmp, "input")
            sub = os.path.join(inp_dir, "shotvault")
            os.makedirs(sub, exist_ok=True)
            img1 = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
            img1.save(os.path.join(sub, "glob.png"))
            img2 = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
            img2.save(os.path.join(sub, "s2.png"))

            vid_p = os.path.join(sub, "s2_ref.mp4")
            qc_mod.write_mp4_with_pcm(torch.rand((24, 64, 64, 3)), torch.ones((2, 48000)) * 0.2, 48000, 24, vid_p)
            wav_path = os.path.join(sub, "guide.wav")
            import av  # type: ignore
            with av.open(wav_path, mode="w") as c:
                st = c.add_stream("pcm_s16le", rate=32000)
                pcm = (torch.sin(torch.linspace(0, 60, 32000)) * 8000).to(torch.int16).unsqueeze(0).repeat(2, 1).numpy()
                fr = av.AudioFrame.from_ndarray(pcm, format="s16p", layout="stereo")
                fr.rate = 32000
                for p in st.encode(fr):
                    c.mux(p)
                for p in st.encode(None):
                    c.mux(p)

            dplan = {
                "version": "5.0", "seed_video_path": "", "seed_trim_end_sec": 0.0,
                "global_ref_images": ["shotvault/glob.png"], "global_ref_audios": [],
                "shots": [
                    {"shot_index": 1, "duration_sec": 2.0, "shot_type": 1, "hard_cut": False, "inherit_global_refs": True,
                     "prompt": "<Picture 1> 出招", "ref_images": [], "ref_videos": [], "ref_audios": [], "guide_audio": ""},
                    {"shot_index": 2, "duration_sec": 2.0, "shot_type": 0, "hard_cut": False, "inherit_global_refs": True,
                     "prompt": "<Picture 1> <Picture 2> <Video 1> 对话", "ref_images": ["shotvault/s2.png"],
                     "ref_videos": [{"path": "shotvault/s2_ref.mp4", "with_audio": False}], "ref_audios": [], "guide_audio": "shotvault/guide.wav"},
                ],
            }
            with patch("folder_paths.get_output_directory", return_value=tmp), patch("folder_paths.get_input_directory", return_value=inp_dir):
                pipe0, total, _ = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]().plan(
                    "E2E_MM", "auto", "", "", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22",
                    director_plan_json=json.dumps(dplan),
                )
                self.assertEqual(total, 2)
                load_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
                cond_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultH3Conditioning"]()
                clip = _CaptureClip()
                p2_res = load_n.load_shot(pipe0, "manual_picks", "2", "auto_latent_first", "fixed", 1, "", 0.0)
                p2 = p2_res[0]
                self.assertEqual(len(p2["loaded_ref_images"]), 2)
                self.assertEqual(len(p2["loaded_ref_videos"]), 1)
                self.assertGreater(p2["shot_audio_guide"]["waveform"].shape[-1], 800)
                self.assertEqual(p2["shot_audio_guide"]["waveform"].shape[-1] % 800, 0)
                pos2, _, _ = cond_n.encode_shot(clip, p2, "match", vae=_MockVAE(), audio_vae=_MockAudioVAE())
                self.assertEqual(len(pos2[0][1]["minimax_refs"]), 3)

    def test_server_routes_upload_parse_preview_pre_run_state_and_security(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp_dir = os.path.join(tmp, "input")
            os.makedirs(inp_dir, exist_ok=True)
            im_path = os.path.join(tmp, "sample.png")
            Image.fromarray(np.zeros((48, 80, 3), dtype=np.uint8)).save(im_path)
            with open(im_path, "rb") as f:
                up = srv_mod.handle_upload_bytes(f.read(), "sample.png", input_root=inp_dir)
            self.assertTrue(up["ok"])
            self.assertEqual((up["width"], up["height"]), (80, 48))
            resolved = srv_mod.resolve_input_media_path(up["rel_path"], input_root=inp_dir)
            self.assertTrue(os.path.isfile(resolved))
            with self.assertRaises(ValueError):
                srv_mod.resolve_input_media_path("../../etc/passwd", input_root=inp_dir)

            prev = srv_mod.handle_parse_preview({"script_text": "& 4.0 &\nA\n===\n& 5.0 &\n@type=文戏\nB"})
            self.assertEqual(len(prev["shots"]), 2)
            self.assertEqual(prev["shots"][1]["shot_type"], 0)

            with patch("folder_paths.get_output_directory", return_value=tmp):
                st = srv_mod.handle_sync_plan({
                    "project": "PreRunProj", "fps": 24,
                    "director_plan_json": json.dumps({"shots": prev["shots"]}),
                })
                self.assertEqual([s["status"] for s in st["shots"]], ["UNRENDERED", "UNRENDERED"])

    def test_director_panel_js_syntax_and_contracts(self):
        for rel in ("web/director_panel.js", "web/sidebar_tab.js"):
            p = os.path.join(PKG_DIR, rel)
            proc = subprocess.run(["node", "--check", p], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(os.path.join(PKG_DIR, "web", "director_panel.js"), "r", encoding="utf-8") as f:
            js = f.read()
        for tok in ("addDOMWidget", "director_plan_json", "/shotvault/upload", "/shotvault/parse-preview", "/shotvault/sync-plan", "ShotVaultLoad", "wheel"):
            self.assertIn(tok, js)

    def test_5_v5_workflows_dual_hemisphere_layout_zero_aabb_overlap_and_widgets(self):
        wf_dir = os.path.join(PKG_DIR, "example_workflows")
        wf_files = sorted(fn for fn in os.listdir(wf_dir) if fn.endswith(".json"))
        self.assertEqual(len(wf_files), 5)
        for fn in wf_files:
            with open(os.path.join(wf_dir, fn), "r", encoding="utf-8") as f:
                wf = json.load(f)
            nodes = wf["nodes"]
            types = {n["type"] for n in nodes}
            self.assertIn("ShotVaultH3Conditioning", types)
            self.assertGreaterEqual(len(wf.get("groups", [])), 6)
            for i in range(len(nodes)):
                a = nodes[i]
                ax1, ay1 = a["pos"]
                ax2, ay2 = ax1 + a["size"][0], ay1 + a["size"][1]
                for j in range(i + 1, len(nodes)):
                    b = nodes[j]
                    bx1, by1 = b["pos"]
                    bx2, by2 = bx1 + b["size"][0], by1 + b["size"][1]
                    self.assertTrue(
                        ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1,
                        f"{fn}: overlapping nodes #{a['id']}({a['type']}) and #{b['id']}({b['type']})",
                    )


if __name__ == "__main__":
    unittest.main()
