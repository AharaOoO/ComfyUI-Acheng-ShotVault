"""Phase 4 audit regression tests: real ComfyUI Sampler/CFGGuider/H3ContextWindowGuider + P0/P1/P2 fixes."""

import importlib
import json
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
cw_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.context_window_guider")
grid_math = importlib.import_module("ComfyUI-Universal-ShotVault.core.grid_math")
asset_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.asset_and_turnaround")
dag_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.lineage_dag")
srv_mod = importlib.import_module("ComfyUI-Universal-ShotVault.server_routes")
vs_mod = importlib.import_module("ComfyUI-Universal-ShotVault.core.vault_store")

import comfy.model_base  # type: ignore
import comfy.model_patcher  # type: ignore
import comfy.nested_tensor  # type: ignore
import comfy.samplers  # type: ignore
import comfy.supported_models  # type: ignore
import comfy.utils  # type: ignore
from comfy_extras import nodes_custom_sampler  # type: ignore


class _TinyH3DiffModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.patch_size = (1, 2, 2)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.recorded_calls = []

    def preprocess_text_embeds(self, x):
        return x

    def forward(self, xc, t, context=None, transformer_options=None, minimax_payload=None, **kwargs):
        import comfy.patcher_extension  # type: ignore

        topts = transformer_options or {}
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, topts),
        ).execute(xc, t, context, topts, minimax_payload=minimax_payload, **kwargs)

    def _forward(self, xc, t, context=None, transformer_options=None, minimax_payload=None, **kwargs):
        sigma_est = float(t.flatten()[0].item()) / 1000.0
        kfs = list((minimax_payload or {}).get("keyframes") or ())
        is_seq = isinstance(xc, (list, tuple))
        x0 = xc[0] if is_seq else xc
        self.recorded_calls.append({
            "sigma": sigma_est,
            "x_shape": tuple(x0.shape),
            "kf_tags": [(kf.get("shotvault_continuation", False), kf.get("is_window_anchor", False), kf.get("audio_latent") is not None) for kf in kfs],
            "cond_v_norms": [float(v.abs().sum().item()) for v in ((minimax_payload or {}).get("cond_video_latents") or ())],
        })
        if is_seq:
            return [(s * 0.15 + 0.05).contiguous() for s in xc]
        return (xc * 0.15 + 0.05).contiguous()


def _make_real_h3_model(with_av_shift: bool = True):
    cfg = comfy.supported_models.MiniMaxH3({"image_model": "minimax_h3", "disable_unet_model_creation": True})
    if not with_av_shift:
        cfg.sampling_settings = {"shift": 12.0}
    base = comfy.model_base.MiniMaxH3(cfg, device=torch.device("cpu"))
    if not with_av_shift:
        # Force audio_scale() == 1.0 to test partial audio clock rebase
        object.__setattr__(base, "audio_scale", lambda: 1.0)
    base.diffusion_model = _TinyH3DiffModule()
    base.model_config.unet_config["disable_unet_model_creation"] = True
    patcher = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    return patcher, base.diffusion_model


class _MockVAE:
    def encode(self, frames):
        f = int(frames.shape[0])
        h, w = int(frames.shape[1]) // 16, int(frames.shape[2]) // 16
        return torch.ones((1, 24, grid_math.video_latent_t(max(5, f)), max(1, h), max(1, w)), dtype=torch.float32) * 0.7


class _MockAudioVAE:
    audio_sample_rate = 32000

    def encode(self, wav):
        t_40 = max(1, round(int(wav.shape[1]) / 32000.0 * 40.0))
        return torch.ones((1, 32, 2, t_40), dtype=torch.float32) * 0.4


def _av_latent(frames: int, h_lat: int = 2, w_lat: int = 2):
    vt = grid_math.video_latent_t(frames)
    at = grid_math.audio_latent_t(frames)
    return {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, vt, h_lat, w_lat)), torch.zeros((1, 32, 2, at))))}


class TestPhase4AuditRegressions(unittest.TestCase):
    def test_p0_1_real_cfg_guider_single_window_negative_none_sampler_runs(self):
        patcher, diff_mod = _make_real_h3_model()
        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        lat = _av_latent(124)
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        guider, out_lat = wg.prepare_guider(patcher, pos, lat, "single_or_1st", "auto_over_192f", 192, 39, "22+5", 0.85, 7, True, False, 1.0)
        sampler = comfy.samplers.ksampler("euler")
        noise = nodes_custom_sampler.RandomNoise.execute(123).result[0]
        sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
        res = nodes_custom_sampler.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, out_lat).result[0]
        v_out, _ = res["samples"].unbind()
        self.assertEqual(tuple(v_out.shape), (1, 24, grid_math.video_latent_t(124), 2, 2))
        self.assertGreaterEqual(len(diff_mod.recorded_calls), 2)

    def test_p0_1_real_cfg_guider_single_window_with_negative_cfg(self):
        patcher, diff_mod = _make_real_h3_model()
        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        neg = [[torch.zeros((1, 4, 8), dtype=torch.float32), {}]]
        guider, out_lat = wg.prepare_guider(patcher, pos, _av_latent(124), "single_or_1st", "off", 192, 39, "22+5", 0.85, 7, True, False, 2.5, negative=neg)
        res = nodes_custom_sampler.SamplerCustomAdvanced.execute(
            nodes_custom_sampler.RandomNoise.execute(5).result[0], guider, comfy.samplers.ksampler("euler"),
            torch.tensor([1.0, 0.0], dtype=torch.float32), out_lat
        ).result[0]
        self.assertEqual(len(diff_mod.recorded_calls), 2)  # positive + negative in 1 CFG step
        self.assertTrue(torch.isfinite(res["samples"].unbind()[0]).all())

    def test_p0_1_real_multi_window_guider_226f_sampler_and_live_x0_anchoring(self):
        patcher, _ = _make_real_h3_model()
        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        guider, out_lat = wg.prepare_guider(
            patcher, pos, _av_latent(226), "single_or_1st", "force_on", 192, 39, "22+5", 0.85, 7, True, False, 1.0,
            vae=_MockVAE(), audio_vae=_MockAudioVAE(),
        )
        res = nodes_custom_sampler.SamplerCustomAdvanced.execute(
            nodes_custom_sampler.RandomNoise.execute(9).result[0], guider, comfy.samplers.ksampler("euler"),
            torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32), out_lat
        ).result[0]
        v_out, _ = res["samples"].unbind()
        self.assertEqual(v_out.shape[2], grid_math.video_latent_t(226))
        self.assertTrue(any(abs(a.abs().sum().item()) > 0 for a in guider.anchors[1]))

    def test_p0_1_real_multi_window_guider_396f_3_windows_and_keyframe_until_strip(self):
        patcher, diff_mod = _make_real_h3_model()
        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        pipe = {"is_continuation": True, "overlap_frames": 22}
        tapered = torch.ones((22, 32, 32, 3), dtype=torch.float32) * 0.5
        prev_aud = {"waveform": torch.ones((1, 2, 37 * 800)) * 0.1, "sample_rate": 32000}
        guider, out_lat = wg.prepare_guider(
            patcher, pos, _av_latent(396), "single_or_1st", "auto_over_192f", 192, 39, "22+5", 0.85, 7, True, False, 1.0,
            vae=_MockVAE(), audio_vae=_MockAudioVAE(), tapered_prev_images=tapered, prev_audio=prev_aud, shot_pipe=pipe,
        )
        self.assertGreaterEqual(len(guider.window_set.windows), 3)
        nodes_custom_sampler.SamplerCustomAdvanced.execute(
            nodes_custom_sampler.RandomNoise.execute(11).result[0], guider, comfy.samplers.ksampler("euler"),
            torch.tensor([0.95, 0.50, 0.0], dtype=torch.float32), out_lat
        )
        high_calls = [c for c in diff_mod.recorded_calls if c["sigma"] > 0.90]
        low_calls = [c for c in diff_mod.recorded_calls if c["sigma"] < 0.60]
        self.assertTrue(any(any(t[0] for t in c["kf_tags"]) for c in high_calls))
        self.assertTrue(all(not any(t[0] for t in c["kf_tags"]) for c in low_calls))

    def test_p0_2_assemble_defer_until_complete_skips_midrun_and_runs_at_end(self):
        plan_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        save_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()
        asm_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultAssemble"]()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("folder_paths.get_output_directory", return_value=tmp):
                pipe0, _, _ = plan_n.plan("DeferProj", "auto", "c", "& 1.5 &\na\n===\n& 1.5 &\nb", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22")
                p1, *_ = load_n.load_shot(pipe0, "sequence_auto", "1", "auto_latent_first", "fixed", 1, "", 0.0)
                sv1 = save_n.save_take(torch.rand((36, 64, 64, 3)), p1, auto_queue_next=False, crf=20, codec="h264")
                m1, _, rep1 = asm_n.assemble("DeferProj", "defer_until_complete", "auto_prefer_reference", 15.0, 0.0, True, True, shot_pipe=sv1["result"][2])["result"]
                self.assertEqual(m1, "")
                self.assertEqual(json.loads(rep1)["status"], "deferred")

                p2, *_ = load_n.load_shot(pipe0, "sequence_auto", "1", "auto_latent_first", "fixed", 2, "", 0.0)
                sv2 = save_n.save_take(torch.rand((36, 64, 64, 3)), p2, auto_queue_next=False, crf=20, codec="h264")
                m2, fc2, _ = asm_n.assemble("DeferProj", "defer_until_complete", "auto_prefer_reference", 15.0, 0.0, True, True, shot_pipe=sv2["result"][2])["result"]
                self.assertTrue(os.path.isfile(m2))
                self.assertTrue(os.path.isfile(fc2))

    def test_p0_3_on_user_prompt_resets_abort_and_scoped_cancel_ignores_foreign(self):
        srv_mod.register_shotvault_prompt_id("sv-pid-1")
        self.assertTrue(srv_mod.handle_cancel_request({"prompt_id": "sv-pid-1"})["aborted"])
        self.assertFalse(srv_mod.handle_cancel_request({"prompt_id": "foreign-pid-999"})["matched"])
        # Non-ShotVault graph keeps abort flag
        srv_mod._on_user_prompt({"prompt": {"1": {"class_type": "KSampler"}}, "prompt_id": "other"})
        self.assertTrue(srv_mod.is_aborted())
        # ShotVault graph on main UI resets abort flag automatically!
        srv_mod._on_user_prompt({"prompt": {"1": {"class_type": "ShotVaultPlan"}}, "prompt_id": "sv-pid-2"})
        self.assertFalse(srv_mod.is_aborted())

    def test_p1_4_save_omits_tail_latent_when_tail_trimmed_and_keeps_when_untrimmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = vs_mod.VaultStore(tmp)
            lat = _av_latent(39, h_lat=2, w_lat=2)  # gen_frames = 39
            m_trim = store.save_take_context("t_trim", 1, torch.rand((28, 32, 32, 3)), None, 48000, 24, "h", "", "1", 28, 0, 1, sampled_latent=lat, raw_trimmed_frames=28)
            self.assertFalse(m_trim["has_tail_latent"])
            self.assertEqual(m_trim["tail_latent_skipped_reason"], "tail_trimmed")
            m_exact = store.save_take_context("t_exact", 1, torch.rand((39, 32, 32, 3)), None, 48000, 24, "h", "", "1", 39, 0, 1, sampled_latent=lat, raw_trimmed_frames=39)
            self.assertTrue(m_exact["has_tail_latent"])

    def test_p1_5_continuation_audio_end_aligned_zero_early_shift(self):
        plan_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        save_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("folder_paths.get_output_directory", return_value=tmp):
                pipe0, _, _ = plan_n.plan("AudEnd", "auto", "c", "& 3.0 &\na\n===\n& 2.0 &\nb", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22")
                p1, *_ = load_n.load_shot(pipe0, "manual_picks", "1", "pixel_clean_taper", "fixed", 1, "", 0.0)
                sr = 48000
                # Silence everywhere except final 100ms pulse right at seam!
                wav = torch.zeros((1, 2, 72 * 2000), dtype=torch.float32)
                wav[:, :, -4800:] = 0.8
                save_n.save_take(torch.rand((72, 64, 64, 3)), p1, False, 20, "h264", final_audio={"waveform": wav, "sample_rate": sr})

                p2, *_ = load_n.load_shot(pipe0, "manual_picks", "2", "pixel_clean_taper", "fixed", 1, "", 0.0)
                pa = p2["prev_audio"]["waveform"]
                # 1114 continuation audio contract: 22f gear takes [-22f : -2f] = 20f (26,667 samples @ 32kHz), dropping final 2f pop guard
                self.assertEqual(pa.shape[-1], round(20.0 / 24.0 * 32000))
                self.assertGreater(float(pa[:, :, -800:].abs().max().item()), 0.3)
                self.assertLess(float(pa[:, :, :1600].abs().max().item()), 0.05)

    def test_p1_6_speech_spans_shifted_by_trim_head_frames(self):
        plan_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("folder_paths.get_output_directory", return_value=tmp):
                script = "& 3.0 &\na\n===\n& 3.0 &\n[00:01.0-00:02.0] <d>说话测试</d>"
                pipe0, _, _ = plan_n.plan("SpShift", "auto", "c", script, "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22")
                p1, *_ = load_n.load_shot(pipe0, "manual_picks", "1", "pixel_clean_taper", "fixed", 1, "", 0.0)
                sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]().save_take(torch.rand((72, 64, 64, 3)), p1, False, 20, "h264")
                p2, *_ = load_n.load_shot(pipe0, "manual_picks", "2", "pixel_clean_taper", "fixed", 1, "", 0.0)
                self.assertEqual(p2["trim_head_frames"], 22)
                self.assertEqual(p2["speech_spans"], [(24 + 22, 48 + 22)])

    def test_p1_7_cross_window_slice_preserves_negative_resolved_frame_index(self):
        w1 = cw_mod.Window(153, 192, 45, 57, 255, 320)
        kf = [{"resolved_frame_index": 140, "latent": torch.ones((1, 24, 12, 2, 2))}]  # starts at 140 (before 153) covering 39f through 179
        sliced = cw_mod.slice_keyframes_for_window(kf, w1)
        self.assertEqual(len(sliced), 1)
        # First 5-token chunk spans 140..157, starting 13 frames BEFORE w1.frame_start (153) -> -13 preserved!
        self.assertEqual(sliced[0]["resolved_frame_index"], -13)

    def test_p1_8_asymmetric_2nd_refine_uses_5_frames_and_locks_audio_mask(self):
        patcher, _ = _make_real_h3_model()
        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        lat = _av_latent(124)
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        pipe = {"is_continuation": True, "overlap_frames": 22}
        tapered = torch.ones((22, 32, 32, 3), dtype=torch.float32) * 0.6
        aud_guide = {"waveform": torch.ones((1, 2, 32000)) * 0.2, "sample_rate": 32000}
        guider, out_lat = wg.prepare_guider(
            patcher, pos, lat, "2nd_refine", "off", 192, 39, "22+5", 0.85, 7, True, True, 1.0,
            vae=_MockVAE(), audio_vae=_MockAudioVAE(), tapered_prev_images=tapered, shot_audio_guide=aud_guide, shot_pipe=pipe,
        )
        kfs = guider.original_conds["positive"][0]["minimax_keyframes"]
        v_kf = [k for k in kfs if k.get("latent") is not None][0]
        self.assertEqual(v_kf["latent"].shape[2], 2)  # 5 frames == 2 latent steps!
        self.assertEqual(v_kf["resolved_frame_index"], 22 - 5)
        vm, am = out_lat["noise_mask"].unbind()
        self.assertTrue((vm == 1).all() and (am == 0).all())

    def test_p1_9_keep_downstream_on_reroll_is_scoped_not_permanent(self):
        plan_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]()
        load_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]()
        save_n = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultSave"]()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("folder_paths.get_output_directory", return_value=tmp):
                pipe0, _, _ = plan_n.plan("ScopeLock", "auto", "c", "& 2.0 &\na\n===\n& 2.0 &\nb\n===\n& 2.0 &\nc", "16:9 (1344x768)", 0.25, 24, "cumulative_cursor", "22")
                for idx in (1, 2, 3):
                    p, *_ = load_n.load_shot(pipe0, "manual_picks", str(idx), "pixel_clean_taper", "fixed", idx, "", 0.0)
                    save_n.save_take(torch.rand((p["target_frames"], 64, 64, 3)), p, False, 20, "h264")
                # Reroll Shot 1 WITH keep_downstream_on_reroll=True -> Shot 2 becomes LOCKED_KEEP (never permanently locked_keep!)
                p1_rr, *_ = load_n.load_shot(pipe0, "manual_picks", "1", "pixel_clean_taper", "fixed", 99, "", 0.0, keep_downstream_on_reroll=True)
                save_n.save_take(torch.rand((48, 64, 64, 3)), p1_rr, False, 20, "h264")
                store = vs_mod.VaultStore(pipe0["project_root"])
                proj = store.load_project()
                self.assertFalse(proj["shots"]["2"]["locked_keep"])
                self.assertFalse(proj["shots"]["3"]["locked_keep"])
                dag = dag_mod.LineageDAG.evaluate(pipe0["shots"], proj)
                self.assertEqual([d["status"] for d in dag], ["VALID", "LOCKED_KEEP", "VALID"])
                # Reroll Shot 1 WITHOUT keep_downstream -> Shot 2 goes back to STALE cleanly!
                p1_strict, *_ = load_n.load_shot(pipe0, "manual_picks", "1", "pixel_clean_taper", "fixed", 101, "", 0.0, keep_downstream_on_reroll=False)
                save_n.save_take(torch.rand((48, 64, 64, 3)), p1_strict, False, 20, "h264")
                dag2 = dag_mod.LineageDAG.evaluate(pipe0["shots"], store.load_project())
                self.assertEqual([d["status"] for d in dag2], ["VALID", "STALE", "VALID"])

    def test_p2_a1_real_turnaround_splits_hanli_and_zhaoyun_and_load_wires_assets(self):
        from PIL import Image  # type: ignore
        import shutil

        inp = os.path.join(COMFY_ROOT, "input")
        hanli = os.path.join(inp, "02_韩立_脸+四视图.png")
        zhaoyun = os.path.join(inp, "02_CHAR_ZHAOYUN_4view.png")
        for sheet in (hanli, zhaoyun):
            if os.path.isfile(sheet):
                arr = torch.from_numpy(__import__("numpy").asarray(Image.open(sheet).convert("RGB"), dtype="float32") / 255.0)
                panels = asset_mod.split_white_turnaround(arr, layout="auto")
                self.assertEqual(len(panels), 4)
                for p in panels:
                    self.assertGreater(p.shape[0], 900)  # Full height, never horizontal 2x2 waist chop!

        if os.path.isfile(hanli):
            with tempfile.TemporaryDirectory() as tmp:
                assets_dir = os.path.join(tmp, "assets")
                os.makedirs(os.path.join(assets_dir, "chars"), exist_ok=True)
                shutil.copy(hanli, os.path.join(assets_dir, "chars", "韩立.png"))
                with patch("folder_paths.get_output_directory", return_value=tmp):
                    pipe0, _, _ = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultPlan"]().plan(
                        "TurnProj", "auto", "c", "& 2.0 &\n韩立 拔剑对峙", "16:9 (1344x768)", 0.25, 24,
                        "cumulative_cursor", "22", assets_root_dir=assets_dir, auto_split_turnaround=True,
                    )
                    res = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultLoad"]().load_shot(pipe0, "manual_picks", "1", "hard_cut", "fixed", 1, "", 0.0)
                    a1, a2 = res[8], res[9]
                    self.assertGreater(a1.shape[1], 900)
                    self.assertGreater(a2.shape[1], 900)
                    self.assertNotEqual(a1.shape[2], a2.shape[2])  # Closeup vs Full-body cropped widths!

    def test_p2_a2_partial_2nd_pass_audio_rebase_when_audio_scale_is_1(self):
        patcher, diff_mod = _make_real_h3_model(with_av_shift=False)
        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        lat = _av_latent(124)
        v0, a0 = lat["samples"].unbind()
        lat = {"samples": comfy.nested_tensor.NestedTensor((v0 + 0.5, a0 + 0.5))}
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        guider, out_lat = wg.prepare_guider(patcher, pos, lat, "2nd_refine", "off", 192, 39, "22+5", 0.0, 7, True, False, 1.0)
        res = nodes_custom_sampler.SamplerCustomAdvanced.execute(
            nodes_custom_sampler.RandomNoise.execute(33).result[0], guider, comfy.samplers.ksampler("euler"),
            torch.tensor([0.815, 0.0], dtype=torch.float32), out_lat
        ).result[0]
        self.assertEqual(len(diff_mod.recorded_calls), 1)
        self.assertGreater(diff_mod.recorded_calls[0]["x_shape"][1], 0)
        self.assertTrue(torch.isfinite(res["samples"].unbind()[1]).all())


if __name__ == "__main__":
    unittest.main()
