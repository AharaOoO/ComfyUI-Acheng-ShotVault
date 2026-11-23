"""Phase 3 tests: 6 nodes, 10-port ShotVaultLoad, auto-trim, #749 compatibility, clean/final isolation, 5 workflows."""

import importlib
import json
import math
import os
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

sv_pkg = importlib.import_module("ComfyUI-Universal-ShotVault")
taper_pkg = importlib.import_module("ComfyUI-H3-Context-Noise")
grid_math = importlib.import_module("ComfyUI-Universal-ShotVault.core.grid_math")
vs_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.vault_store")
import comfy.nested_tensor  # type: ignore


class _MockVAE:
    def encode(self, frames):
        f = int(frames.shape[0])
        h, w = int(frames.shape[1]) // 16, int(frames.shape[2]) // 16
        return torch.ones((1, 24, grid_math.video_latent_t(max(5, f)), max(1, h), max(1, w)), dtype=torch.float32) * 0.5


class _MockAudioVAE:
    audio_sample_rate = 32000

    def encode(self, wav):
        t_40 = max(1, round(int(wav.shape[1]) / 32000.0 * 40.0))
        return torch.ones((1, 32, 2, t_40), dtype=torch.float32) * 0.25


class TestPhase3NodesAndWorkflows(unittest.TestCase):
    def test_registry_and_port0_alignment(self):
        mappings = sv_pkg.NODE_CLASS_MAPPINGS
        expected = {
            "ShotVaultPlan", "ShotVaultLoad", "ShotVaultLoopDispatch", "ShotVaultH3Conditioning", "ShotVaultWindowGuider",
            "ShotVaultSeamTrim", "ShotVaultSave", "ShotVaultAssemble",
        }
        self.assertEqual(set(mappings.keys()), expected)
        self.assertEqual(sv_pkg.WEB_DIRECTORY, "./web")

        st_in = list(mappings["ShotVaultSeamTrim"].INPUT_TYPES()["required"].items())[0]
        self.assertEqual(st_in[1][0], "IMAGE")
        self.assertEqual(mappings["ShotVaultSeamTrim"].RETURN_TYPES[0], "IMAGE")
        sv_in = list(mappings["ShotVaultSave"].INPUT_TYPES()["required"].items())[0]
        self.assertEqual(sv_in[1][0], "IMAGE")
        self.assertEqual(mappings["ShotVaultSave"].RETURN_TYPES[0], "IMAGE")

    def test_load_outputs_10_ports_and_no_seed_widget_shadow(self):
        load_cls = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]
        req = load_cls.INPUT_TYPES()["required"]
        self.assertNotIn("seed", req)
        self.assertIn("shot_seed", req)
        self.assertEqual(req["av_sync_lead_sec"][1]["default"], 0.00)
        self.assertEqual(len(load_cls.RETURN_TYPES), 10)
        self.assertTrue(math.isnan(load_cls.IS_CHANGED(
            {"project_root": ""}, "sequence_auto", "1", "auto_latent_first", "randomize", 1, "", 0.0
        )))
        h1 = load_cls.IS_CHANGED({"project_root": "p"}, "manual_picks", "1", "auto_latent_first", "fixed", 7, "", 0.0)
        h2 = load_cls.IS_CHANGED({"project_root": "p"}, "manual_picks", "1", "auto_latent_first", "fixed", 7, "", 0.0)
        self.assertEqual(h1, h2)

    def test_seam_trim_negative1_auto_resolves_from_pipe(self):
        trim_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSeamTrim"]()
        imgs = torch.rand((39, 32, 32, 3))
        pipe = {"fps": 24, "trim_head_frames": 22, "keep_frames": 10, "is_continuation": False}
        out_img, _ = trim_n.trim_and_match(imgs, -1, -1, "off", 12, "0.75", shot_pipe=pipe)
        self.assertEqual(out_img.shape[0], 10)

    def test_e2e_2shot_pipeline_with_749_and_clean_upscale_isolation(self):
        mappings = sv_pkg.NODE_CLASS_MAPPINGS
        plan_n = mappings["ShotVaultPlan"]()
        load_n = mappings["ShotVaultLoad"]()
        trim_n = mappings["ShotVaultSeamTrim"]()
        save_n = mappings["ShotVaultSave"]()
        asm_n = mappings["ShotVaultAssemble"]()
        taper_n = taper_pkg.NODE_CLASS_MAPPINGS["MiniMaxH3ContextTaperNoise"]()

        with tempfile.TemporaryDirectory() as tmp:
            script = "& 2.0 &\n拔刀蓄势\n===\n& 2.0 &\n@type=文戏\n静默收剑"
            with patch("folder_paths.get_output_directory", return_value=tmp):
                pipe0, total, _ = plan_n.plan(
                    project_name="SmokeFilm", script_format="auto", common_prompt="写实质感",
                    script_text=script, aspect_ratio="16:9 (1344x768)", megapixels=0.25, fps=24,
                    duration_mode="cumulative_cursor", context_frames="22",
                )
                self.assertEqual(total, 2)

                res1 = load_n.load_shot(pipe0, "manual_picks", "1", "auto_latent_first", "custom_string", 1, "18446744073709551615", 0.00)
                pipe1, prev_img1, is_drama1, _pr1, _w1, _h1, tgt1, seed1, _ar1, _ar2 = res1
                self.assertFalse(is_drama1)
                self.assertEqual(seed1, 18446744073709551615)
                tapered1, _ = taper_n.inject(prev_img1, tail_frames=22, alpha=0.45, alpha_end=0.10, ramp_frames=3, seed=seed1 & 0xFFFFFFFF)
                self.assertEqual(tapered1.shape[0], 1)

                t_steps1 = grid_math.video_latent_t(tgt1)
                lat1 = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, t_steps1, 4, 4)), torch.zeros((1, 32, 2, grid_math.audio_latent_t(tgt1)))))}
                raw_clean1 = torch.rand((tgt1, 64, 64, 3))
                raw_aud1 = {"waveform": torch.sin(torch.linspace(0, 20, round(tgt1 / 24 * 48000))).repeat(1, 2, 1) * 0.1, "sample_rate": 48000}
                clean_trim1, aud_trim1 = trim_n.trim_and_match(raw_clean1, -1, -1, "seam_12f_mkl_lab", 12, "0.75, 0.45, 0.18", audio=raw_aud1, shot_pipe=pipe1)
                final_up1 = clean_trim1.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

                sv1 = save_n.save_take(final_up1, pipe1, auto_queue_next=False, crf=20, codec="h264", final_audio=aud_trim1, clean_images=clean_trim1, sampled_latent=lat1)
                tid1 = sv1["result"][3]
                store = vs_mod.VaultStore(pipe1["project_root"])
                ctx1 = store.load_take_context(tid1)
                self.assertEqual(tuple(ctx1["tensors"]["clean_tail"].shape[1:3]), (64, 64))
                self.assertFalse(ctx1["meta"]["has_tail_latent"])  # Tail trimmed (48 keep < 56 gen) -> safely skipped!

                res2 = load_n.load_shot(pipe0, "manual_picks", "2", "auto_latent_first", "fixed", 7, "", 0.00)
                pipe2, prev_img2, is_drama2, _pr2, _w2, _h2, tgt2, _s2, *_ = res2
                self.assertTrue(is_drama2)
                self.assertEqual(tuple(prev_img2.shape[1:3]), (64, 64))

                raw_clean2 = torch.rand((tgt2, 64, 64, 3))
                raw_aud2 = {"waveform": torch.sin(torch.linspace(0, 20, round(tgt2 / 24 * 48000))).repeat(1, 2, 1) * 0.1, "sample_rate": 48000}
                clean_trim2, aud_trim2 = trim_n.trim_and_match(raw_clean2, -1, -1, "seam_12f_mkl_lab", 12, "0.75, 0.45, 0.18", audio=raw_aud2, shot_pipe=pipe2)
                final_up2 = clean_trim2.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
                sv2 = save_n.save_take(final_up2, pipe2, auto_queue_next=False, crf=20, codec="h264", final_audio=aud_trim2, clean_images=clean_trim2)

                asm_res = asm_n.assemble("SmokeFilm", "defer_until_complete", "auto_prefer_reference", 15.0, 0.0, True, True, shot_pipe=sv2["result"][2])
                master_path, fcpxml_path, rep_json = asm_res["result"]
                self.assertTrue(os.path.isfile(master_path))
                self.assertTrue(os.path.isfile(fcpxml_path))
                self.assertEqual(json.loads(rep_json)["total_frames"], pipe1["keep_frames"] + pipe2["keep_frames"])

    def test_5_example_workflows_schema_links_widgets_and_01B_models(self):
        wf_dir = os.path.join(PKG_DIR, "example_workflows")
        wf_files = sorted(fn for fn in os.listdir(wf_dir) if fn.endswith(".json"))
        self.assertEqual(len(wf_files), 5)
        valid_sv_types = set(sv_pkg.NODE_CLASS_MAPPINGS.keys())
        models_root = os.path.join(COMFY_ROOT, "models")
        loader_dirs = {"UNETLoader": "unet", "CLIPLoader": "clip", "VAELoader": "vae", "LoraLoaderModelOnly": "loras", "MiniMaxH3LoRACompatibilityLoaderT8Advanced": "loras", "MiniMaxH3SemanticBridgeConfigT8": "semantic_bridge"}

        for fn in wf_files:
            with open(os.path.join(wf_dir, fn), "r", encoding="utf-8") as f:
                wf = json.load(f)
            nodes_by_id = {n["id"]: n for n in wf["nodes"]}
            links_by_id = {l[0]: l for l in wf["links"]}
            self.assertEqual({n["type"] for n in wf["nodes"] if n["type"] in valid_sv_types}, valid_sv_types)
            for nid, node in nodes_by_id.items():
                if node["type"] == "ShotVaultLoad":
                    self.assertEqual(len(node["widgets_values"]), 9)
                elif node["type"] == "ShotVaultPlan":
                    self.assertEqual(len(node["widgets_values"]), 18)
                elif node["type"] == "ShotVaultLoopDispatch":
                    self.assertEqual(len(node["widgets_values"]), 6)
                for slot_idx, inp in enumerate(node.get("inputs", [])):
                    lid = inp.get("link")
                    if lid is not None:
                        _, s_id, s_slot, d_id, d_slot, _ = links_by_id[lid]
                        self.assertEqual((d_id, d_slot), (nid, slot_idx))
                        self.assertIn(lid, nodes_by_id[s_id]["outputs"][s_slot]["links"])
                if not fn.startswith("01_") and node["type"] in loader_dirs:
                    rel = node["widgets_values"][0]
                    self.assertTrue(os.path.isfile(os.path.join(models_root, loader_dirs[node["type"]], rel)), f"{fn} missing {rel}")

    def test_sidebar_js_non_destructive_and_scoped_handlers(self):
        js_path = os.path.join(PKG_DIR, "web", "sidebar_tab.js")
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
        self.assertIn("activeRefreshFn", js)
        self.assertIn("graphToPrompt", js)
        self.assertIn("prompt_id: pid", js)
        self.assertNotIn('data-tab-id="', js)
        for node_name in sv_pkg.NODE_CLASS_MAPPINGS:
            self.assertTrue(os.path.isfile(os.path.join(PKG_DIR, "web", "docs", f"{node_name}.md")))

    def test_all_workflow_node_types_exist_in_local_comfyui(self):
        import re

        wf_dir = os.path.join(PKG_DIR, "example_workflows")
        used_types: set[str] = set()
        for fn in sorted(os.listdir(wf_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(wf_dir, fn), "r", encoding="utf-8") as f:
                    for n in json.load(f)["nodes"]:
                        t = str(n.get("type") or "")
                        if t and t not in ("Note", "Reroute"):
                            used_types.add(t)

        chunks: list[str] = []
        with open(os.path.join(COMFY_ROOT, "nodes.py"), encoding="utf-8", errors="ignore") as f:
            chunks.append(f.read())
        for sub in ("comfy_extras", "custom_nodes"):
            for dp, dn, fns in os.walk(os.path.join(COMFY_ROOT, sub)):
                dn[:] = [d for d in dn if d not in ("__pycache__", ".git", "tests", "example_workflows")]
                for fn in fns:
                    if fn.endswith(".py"):
                        with open(os.path.join(dp, fn), encoding="utf-8", errors="ignore") as f:
                            chunks.append(f.read())
        joined = "\n".join(chunks)

        missing = [
            t for t in sorted(used_types)
            if not re.search(rf'(?:node_id\s*=\s*["\']{re.escape(t)}["\']|["\']{re.escape(t)}["\']\s*:|\bclass\s+{re.escape(t)}\b)', joined)
        ]
        self.assertEqual(missing, [], f"Unregistered workflow node types: {missing}")


if __name__ == "__main__":
    unittest.main()
