"""Phase 2 tests: ContextWindows anchoring/_late_handover/DP speech, MKL/MaskedBG, Shot 0 ingest, single-pass PCM mux."""

import importlib
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import torch

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_NODES_DIR = os.path.dirname(PKG_DIR)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES_DIR)
for p in (COMFY_ROOT, CUSTOM_NODES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

compat = importlib.import_module("ComfyUI-Acheng-ShotVault.compat")
assembler_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.assembler")
asset_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.asset_and_turnaround")
cw_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.context_window_guider")
color_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.seam_and_color")
qc_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.seam_qc")
seed_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.seed_video_ingest")
vs_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.core.vault_store")

sha256_text = compat.sha256_text
_count_video_frames = assembler_mod._count_video_frames
_decode_mp4_audio_48k = assembler_mod._decode_mp4_audio_48k
assemble_master = assembler_mod.assemble_master
split_white_turnaround = asset_mod.split_white_turnaround
Window = cw_mod.Window
add_window_anchor = cw_mod.add_window_anchor
apply_keyframe_until = cw_mod.apply_keyframe_until
blend = cw_mod.blend
decode_video_chunked = cw_mod.decode_video_chunked
plan_speech_windows = cw_mod.plan_speech_windows
plan_windows = cw_mod.plan_windows
window_weights = cw_mod.window_weights
masked_bg_ycc_match = color_mod.masked_bg_ycc_match
mkl_lab_seam_match = color_mod.mkl_lab_seam_match
trim_av_sync = color_mod.trim_av_sync
build_seam_qc_clip = qc_mod.build_seam_qc_clip
write_mp4_with_pcm = qc_mod.write_mp4_with_pcm
ingest_seed_video = seed_mod.ingest_seed_video
VaultStore = vs_mod.VaultStore


class _FakeModel:
    def __init__(self):
        self.wrappers = {}

    def clone(self):
        m = _FakeModel()
        m.wrappers = {k: dict(v) for k, v in self.wrappers.items()}
        return m

    def add_wrapper_with_key(self, outer, key, fn):
        self.wrappers.setdefault(outer, {})[key] = fn


class TestPhase2GuiderColorMux(unittest.TestCase):
    def test_windows_speech_dp_late_handover_and_chunked_decode(self):
        ws = plan_windows(430, window_frames=192, overlap_frames=39)
        self.assertGreaterEqual(len(ws), 3)
        weights_v = window_weights(ws, "video", late=True)
        ones_parts = [torch.ones((1, 1, w.video_len, 1, 1)) for w in ws]
        ones_out = blend(ones_parts, ws, weights_v, ws[-1].video_end, "video", 2)
        self.assertTrue(torch.allclose(ones_out, torch.ones_like(ones_out), atol=1e-5))

        sp_ws, crossed = plan_speech_windows(396, window_frames=192, overlap_frames=39, speech=[(140, 175), (290, 320)])
        self.assertEqual(crossed, 0)
        self.assertGreaterEqual(len(sp_ws), 2)

        lat = torch.randn((1, 4, 127, 4, 4))

        def mock_vae_decode(z):
            t = int(z.shape[2])
            px = 5 + ((t - 2) // 5) * 17
            out = torch.zeros((px, 4, 4, 3))
            g_cnt = (t - 2) // 5
            for g in range(g_cnt):
                val = z[0, :3, 5 * g, 0, 0]
                s = 17 * g
                e = px if g == g_cnt - 1 else 17 * (g + 1)
                out[s:e] = val.view(1, 1, 1, 3)
            return out

        full_dec = mock_vae_decode(lat)
        chunk_dec = decode_video_chunked(mock_vae_decode, lat, groups_per_chunk=21)
        self.assertEqual(full_dec.shape, chunk_dec.shape)
        self.assertEqual(float((full_dec - chunk_dec).abs().max().item()), 0.0)

    def test_anchor_and_selective_keyframe_until(self):
        w0 = Window(0, 192, 0, 57, 0, 320)
        w1 = Window(153, 192, 45, 57, 255, 320)
        conds = {
            "positive": [{
                "minimax_keyframes": [{"resolved_frame_index": 0, "audio_latent": torch.ones((1, 32, 2, 10))}],
            }]
        }
        ph = add_window_anchor(conds, w0, w1, (1, 24, 57, 4, 4))
        self.assertIsNotNone(ph)
        self.assertEqual(len(conds["positive"][0]["minimax_keyframes"]), 2)

        mdl = apply_keyframe_until(_FakeModel(), until_sigma=0.85)
        import comfy.patcher_extension  # type: ignore

        fn = mdl.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]["shotvault_keyframe_until"]
        captured = {}

        def executor(x, t, ctx, topts, minimax_payload=None, **kw):
            captured["payload"] = minimax_payload
            return x

        payload = {
            "keyframes": [
                {"resolved_frame_index": 0, "latent": torch.ones(1), "shotvault_continuation": True},
                {"resolved_frame_index": 0, "latent": torch.ones(1) * 2, "is_window_anchor": True},
                {"resolved_frame_index": 0, "audio_latent": torch.ones(1) * 3},
            ],
            "refs": [{"latent": torch.ones(1) * 4}],
        }
        fn(executor, torch.zeros(1), torch.tensor([500.0]), None, {}, minimax_payload=payload)
        rem_kfs = captured["payload"]["keyframes"]
        self.assertEqual(len(rem_kfs), 2)
        self.assertFalse(any(k.get("shotvault_continuation") for k in rem_kfs))
        self.assertEqual(len(captured["payload"]["cond_video_latents"]), 2)
        self.assertEqual(len(captured["payload"]["cond_audio_latents"]), 1)

    def test_mkl_lab_masked_bg_and_turnaround_split(self):
        ref = torch.full((6, 64, 64, 3), 0.35)
        ref[..., 0] += 0.05
        clip = torch.full((16, 64, 64, 3), 0.50)
        matched = mkl_lab_seam_match(clip, ref, seam_frames=12)
        self.assertTrue(torch.allclose(matched[12:], clip[12:], atol=1e-5))
        self.assertGreater((matched[0] - clip[0]).abs().mean().item(), 0.01)

        src = torch.full((10, 64, 64, 3), 0.25)
        gen = torch.full((10, 64, 64, 3), 0.33)
        gen[:, 20:44, 20:44, :] = 0.85
        out, rep = masked_bg_ycc_match(gen, src)
        self.assertTrue(rep and rep[0]["applied"])
        self.assertLess((out[:, 2:10, 2:10].mean() - 0.25).abs().item(), 0.04)

        img_trim, aud_trim = trim_av_sync(torch.rand((34, 32, 32, 3)), {"waveform": torch.rand((1, 2, 68000)), "sample_rate": 48000}, trim_head_frames=22, keep_frames=10, fps=24)
        self.assertEqual(img_trim.shape[0], 10)
        self.assertEqual(aud_trim["waveform"].shape[-1], round(10 / 24.0 * 48000))

        sheet = torch.ones((128, 128, 3))
        sheet[10:50, 10:50] = 0.2
        sheet[10:50, 74:114] = 0.4
        sheet[74:114, 10:50] = 0.6
        sheet[74:114, 74:114] = 0.8
        panels = split_white_turnaround(sheet, layout="2x2")
        self.assertEqual(len(panels), 4)
        self.assertLess(panels[0].shape[0], 50)

    def test_seed_ingest_and_single_pass_pcm_master_assemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_mp4 = os.path.join(tmp, "external_seed.mp4")
            src_frames = torch.rand((36, 64, 64, 3))
            src_audio = torch.sin(torch.linspace(0, 60, round(36 / 24 * 48000))).repeat(2, 1) * 0.2
            write_mp4_with_pcm(src_frames, src_audio, 48000, 24, seed_mp4)

            store = VaultStore(os.path.join(tmp, "ShotVault", "MVProj"))
            seed_meta = ingest_seed_video(store, seed_video_path=seed_mp4, seed_trim_end_sec=1.0, width=64, height=64, fps=24)
            self.assertEqual(seed_meta["keep_frames"], 24)

            prev_tid = seed_meta["take_id"]
            plan = []
            for idx in range(1, 5):
                tid, lk = store.reserve_next_take(idx)
                store.release_reserve(lk)
                f_cnt = 18 + idx
                frames = torch.rand((f_cnt, 64, 64, 3))
                wav = torch.sin(torch.linspace(0, 40, round(f_cnt / 24.0 * 48000))).repeat(2, 1) * 0.15
                mp4_rel = f"shots/{tid}.mp4"
                write_mp4_with_pcm(frames, wav, 48000, 24, os.path.join(store.project_root, mp4_rel))
                h = sha256_text(f"p_{idx}")
                store.save_take_context(
                    take_id=tid, shot_index=idx, clean_images=frames, audio_waveform_48k=wav, audio_sr=48000,
                    fps=24, prompt_sha256=h, parent_take_id=prev_tid, seed_text=str(100 + idx),
                    keep_frames=f_cnt, trim_head_frames=22, shot_type=1, video_rel_path=mp4_rel,
                )
                seam_path = build_seam_qc_clip(frames[:6], wav, frames, wav, 24, os.path.join(store.vault_dir, f"seam_{tid}.mp4"))
                self.assertTrue(os.path.isfile(seam_path))
                plan.append({"shot_index": idx, "shot_id": f"shot_{idx:04d}", "is_continuation": True, "prompt_sha256": h})
                prev_tid = tid

            master_mp4, fcpxml_path, _ = assemble_master(
                store, plan, fps=24, stale_policy="block_and_report",
                audio_crossfade_ms=15.0, post_mux_offset_ms=0.0, prepend_seed_video=True, export_fcpxml=True
            )
            self.assertTrue(os.path.isfile(master_mp4))
            self.assertTrue(os.path.isfile(fcpxml_path))

            total_expected_frames = 24 + sum(18 + i for i in range(1, 5))
            v_frames = _count_video_frames(master_mp4)
            self.assertEqual(v_frames, total_expected_frames)
            a_dec = _decode_mp4_audio_48k(master_mp4)
            self.assertIsNotNone(a_dec)
            self.assertLess(abs(v_frames / 24.0 - a_dec.shape[1] / 48000.0), 0.045)

            et = ET.parse(fcpxml_path)
            self.assertEqual(et.getroot().tag, "fcpxml")

    def test_streaming_write_mp4_multi_chunk_over_40_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_mp4 = os.path.join(tmp, "chunked_48f.mp4")
            frames = torch.rand((48, 64, 64, 3), dtype=torch.float32)
            wav = torch.sin(torch.linspace(0, 80, 96000)).repeat(2, 1) * 0.2
            write_mp4_with_pcm(frames, wav, 48000, 24, out_mp4)
            self.assertEqual(_count_video_frames(out_mp4), 48)
            dec = _decode_mp4_audio_48k(out_mp4)
            self.assertIsNotNone(dec)
            self.assertLess(abs(48 / 24.0 - dec.shape[1] / 48000.0), 0.045)

    def test_masked_bg_ycc_skips_full_scene_change(self):
        g = torch.linspace(0.0, 1.0, 64).view(1, 1, 64, 1).expand(6, 64, 64, 3).contiguous()
        src = g.clone()
        gen = (1.0 - g).clone()
        out, rep = masked_bg_ycc_match(gen, src)
        self.assertTrue(rep)
        self.assertFalse(rep[0]["applied"])
        self.assertTrue(torch.allclose(out, gen, atol=1e-6))

    def test_assemble_stale_block_and_report_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultStore(os.path.join(tmp, "StaleAsmProj"))
            frames = torch.rand((24, 64, 64, 3))
            h1, h2 = sha256_text("s1"), sha256_text("s2")
            t1, l1 = store.reserve_next_take(1); store.release_reserve(l1)
            p1 = os.path.join(store.project_root, f"shots/{t1}.mp4")
            write_mp4_with_pcm(frames, None, 48000, 24, p1)
            store.save_take_context(t1, 1, frames, None, 48000, 24, h1, "", "1", 24, 0, 1, video_rel_path=f"shots/{t1}.mp4")

            t2, l2 = store.reserve_next_take(2); store.release_reserve(l2)
            p2 = os.path.join(store.project_root, f"shots/{t2}.mp4")
            write_mp4_with_pcm(frames, None, 48000, 24, p2)
            store.save_take_context(t2, 2, frames, None, 48000, 24, h2, t1, "2", 24, 22, 1, video_rel_path=f"shots/{t2}.mp4")

            # Reroll shot 1 -> shot 2 becomes STALE
            t1b, l1b = store.reserve_next_take(1); store.release_reserve(l1b)
            p1b = os.path.join(store.project_root, f"shots/{t1b}.mp4")
            write_mp4_with_pcm(frames, None, 48000, 24, p1b)
            store.save_take_context(t1b, 1, frames, None, 48000, 24, h1, "", "3", 24, 0, 1, video_rel_path=f"shots/{t1b}.mp4")

            plan = [
                {"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": h1},
                {"shot_index": 2, "shot_id": "shot_0002", "is_continuation": True, "prompt_sha256": h2},
            ]
            with self.assertRaises(ValueError):
                assemble_master(store, plan, fps=24, stale_policy="block_and_report")
            master_mp4, _, rep_json = assemble_master(store, plan, fps=24, stale_policy="allow_with_warning")
            self.assertTrue(os.path.isfile(master_mp4))
            import json
            self.assertEqual(json.loads(rep_json)["warnings"][0]["shot_index"], 2)

    def test_assemble_master_audio_override_and_post_mux_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultStore(os.path.join(tmp, "MasterOvProj"))
            frames = torch.rand((24, 64, 64, 3))
            h1 = sha256_text("m1")
            t1, l1 = store.reserve_next_take(1); store.release_reserve(l1)
            rel = f"shots/{t1}.mp4"
            write_mp4_with_pcm(frames, None, 48000, 24, os.path.join(store.project_root, rel))
            store.save_take_context(t1, 1, frames, None, 48000, 24, h1, "", "1", 24, 0, 1, video_rel_path=rel)
            plan = [{"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": h1}]
            master_wav = torch.sin(torch.linspace(0, 80, 48000)).view(1, 1, -1).repeat(1, 2, 1) * 0.4
            m_mp4, _, _ = assemble_master(
                store, plan, fps=24, stale_policy="block_and_report",
                audio_source_policy="master_override",
                master_audio_override={"waveform": master_wav, "sample_rate": 48000},
                post_mux_offset_ms=100.0, export_fcpxml=False,
            )
            self.assertEqual(_count_video_frames(m_mp4), 24)
            dec = _decode_mp4_audio_48k(m_mp4)
            self.assertIsNotNone(dec)
            self.assertLess(abs(1.0 - dec.shape[1] / 48000.0), 0.045)

    def test_seed_video_30fps_silent_ingest_resamples_to_24fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed30 = os.path.join(tmp, "seed_30fps_silent.mp4")
            write_mp4_with_pcm(torch.rand((30, 64, 64, 3)), None, 48000, 30, seed30)
            store = VaultStore(os.path.join(tmp, "Vault30"))
            meta = ingest_seed_video(store, seed30, seed_trim_end_sec=0.0, width=64, height=64, fps=24)
            self.assertEqual(meta["keep_frames"], 24)
            self.assertIsNotNone(store.load_take_context(meta["take_id"]))

    def test_beat_snap_transient_alignment(self):
        plan_mod = importlib.import_module("ComfyUI-Acheng-ShotVault.nodes.plan_node")
        sr = 48000
        wav = torch.ones((2, sr * 4), dtype=torch.float32) * 0.01
        onset_s = round(2.12 * sr)
        wav[:, onset_s:onset_s + 2400] = 0.9
        snapped = plan_mod._snap_boundaries_to_beats(wav, sr, [2.0, 4.0], max_shift_sec=0.25)
        self.assertGreater(snapped[0], 2.04)
        self.assertLess(abs(snapped[0] - 2.12), 0.06)

    def test_continuation_frames_22_plus_0_skips_2nd_pass_keyframe(self):
        sv_pkg = importlib.import_module("ComfyUI-Acheng-ShotVault")
        import comfy.nested_tensor  # type: ignore

        wg = sv_pkg.NODE_CLASS_MAPPINGS["ShotVaultWindowGuider"]()
        lat = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 17, 2, 2)), torch.zeros((1, 32, 2, 93))))}
        pos = [[torch.ones((1, 4, 8), dtype=torch.float32), {}]]
        tapered = torch.ones((22, 32, 32, 3), dtype=torch.float32) * 0.5
        pipe = {"is_continuation": True, "overlap_frames": 22}

        class _Vae:
            def encode(self, frames):
                f = int(frames.shape[0])
                t = 2 if f <= 5 else ((f - 5) // 17) * 5 + 2
                return torch.ones((1, 24, t, 2, 2))

        p1, _ = wg.prepare_guider(None, pos, lat, "single_or_1st", "off", 192, 39, "22+0", 0.85, 7, True, False, 1.0, vae=_Vae(), tapered_prev_images=tapered, shot_pipe=pipe)
        p2, _ = wg.prepare_guider(None, pos, lat, "2nd_refine", "off", 192, 39, "22+0", 0.85, 7, True, False, 1.0, vae=_Vae(), tapered_prev_images=tapered, shot_pipe=pipe)
        v_kfs_1 = [k for k in (p1[0][1].get("minimax_keyframes") or ()) if k.get("latent") is not None]
        v_kfs_2 = [k for k in (p2[0][1].get("minimax_keyframes") or ()) if k.get("latent") is not None]
        self.assertEqual(len(v_kfs_1), 1)
        self.assertEqual(v_kfs_1[0]["latent"].shape[2], 7)
        self.assertEqual(len(v_kfs_2), 0)


if __name__ == "__main__":
    unittest.main()
