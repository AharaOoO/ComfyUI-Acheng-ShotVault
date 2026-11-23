"""Phase 1 tests: H3 grid math, 800-sample pad, cumulative cursor, parsers, VaultStore, LineageDAG."""

import importlib
import os
import sys
import tempfile
import unittest

import torch

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_NODES_DIR = os.path.dirname(PKG_DIR)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES_DIR)
for p in (COMFY_ROOT, CUSTOM_NODES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

sv_pkg = importlib.import_module("ComfyUI-Universal-ShotVault")
compat = importlib.import_module("ComfyUI-Universal-ShotVault.compat")
grid_math = importlib.import_module("ComfyUI-Universal-ShotVault.core.grid_math")
lineage_dag = importlib.import_module("ComfyUI-Universal-ShotVault.core.lineage_dag")
script_parser = importlib.import_module("ComfyUI-Universal-ShotVault.core.script_parser")
vault_store = importlib.import_module("ComfyUI-Universal-ShotVault.core.vault_store")

sanitize_project_name = compat.sanitize_project_name
sha256_text = compat.sha256_text
align_frame_count = grid_math.align_frame_count
compute_audio_overhang = grid_math.compute_audio_overhang
compute_cumulative_schedule = grid_math.compute_cumulative_schedule
pad_audio_800 = grid_math.pad_audio_800
resolve_audio_guide_end = grid_math.resolve_audio_guide_end
time_shift_sigma = grid_math.time_shift_sigma
validate_tail_slice = grid_math.validate_tail_slice
video_latent_t = grid_math.video_latent_t
LineageDAG = lineage_dag.LineageDAG
parse_script = script_parser.parse_script
parse_shot_numbers = script_parser.parse_shot_numbers
strip_jsonc_comments = script_parser.strip_jsonc_comments
VaultStore = vault_store.VaultStore


class TestPhase1MathParserVault(unittest.TestCase):
    def test_cycle_pos_zero_and_reject_17(self):
        total_t = video_latent_t(align_frame_count(124))
        self.assertEqual(total_t, 37)
        for valid_f in (5, 22, 39, 56):
            start, steps = validate_tail_slice(total_t, valid_f)
            self.assertEqual(start % 5, 0)
            self.assertEqual(steps, video_latent_t(valid_f))
        with self.assertRaises(ValueError):
            validate_tail_slice(total_t, 17)

    def test_audio_800_and_overhang_and_sigma_shift(self):
        wav = torch.zeros((1, 2, 32123), dtype=torch.float32)
        padded = pad_audio_800(wav)
        self.assertEqual(padded.shape[-1] % 800, 0)
        self.assertGreaterEqual(padded.shape[-1], 32123)

        oh = compute_audio_overhang(207, 124)
        self.assertAlmostEqual(oh, 0.333333, places=4)
        self.assertEqual(resolve_audio_guide_end(39, oh), 65)

        s_audio = time_shift_sigma(0.815, shift_from=12.0, shift_to=3.0)
        self.assertGreater(s_audio, 0.0)
        self.assertLess(s_audio, 0.815)

    def test_50_shot_cumulative_cursor_zero_drift(self):
        durs = [2.37] * 50
        sched = compute_cumulative_schedule(
            durs, fps=24, is_continuations=[i > 0 for i in range(50)], context_frames=22, av_sync_lead_sec=0.0
        )
        self.assertEqual(len(sched), 50)
        self.assertEqual(sched[0]["trim_head_frames"], 0)
        self.assertEqual(sum(s["keep_frames"] for s in sched), round(sum(durs) * 24))

        sched_lead = compute_cumulative_schedule([3.0, 3.0], fps=24, context_frames=22, av_sync_lead_sec=1.0)
        self.assertEqual(sched_lead[0]["trim_head_frames"], 24)
        self.assertEqual(sched_lead[1]["trim_head_frames"], 39)

    def test_parsers_unlimited_1114_mv_and_dual_jsonc(self):
        blocks = "\n===\n".join(
            f"& 00:{i*2:02d}.00 - 00:{i*2+1:02d}.50 &\n"
            + ("@cut\n" if i % 10 == 0 and i > 0 else "")
            + ("@type=文戏\n[00:00.2-00:01.1] <d>台词测试</d>\n" if i % 2 == 1 else "高动态格斗招式")
            for i in range(36)
        )
        shots = parse_script(blocks, common_prompt="写实电影光影")
        self.assertEqual(len(shots), 36)
        self.assertFalse(shots[0]["is_continuation"])
        self.assertFalse(shots[10]["is_continuation"])
        self.assertTrue(shots[1]["is_drama_mode"])
        self.assertEqual(len(shots[1]["speech_spans"]), 1)

        s2_jsonc = """
        //**第01集 事件档案**///
        [
          {"id": 1, "时长": "4.5s", "类型": "武戏", "场景衔接": "开场", "url": "http://keep//url"},
          {"id": 2, "时长": 3.0, "类型": "文戏", "场景衔接": "接续",},
        ]
        /* 尾注 */
        """
        s3_jsonc = """
        [
          {"id": 1, "画面提示词": "破空长拳"},
          {"id": 2, "画面提示词": "冷静凝视"},
        ]
        """
        self.assertIn("http://keep//url", strip_jsonc_comments(s2_jsonc))
        jshots = parse_script(s2_jsonc, stage3_json_text=s3_jsonc, script_format="kkdemabu_dual_jsonc")
        self.assertEqual(len(jshots), 2)
        self.assertFalse(jshots[0]["is_continuation"])
        self.assertEqual(jshots[1]["shot_type"], 0)
        self.assertTrue(jshots[1]["is_continuation"])
        self.assertEqual(jshots[1]["local_prompt"], "冷静凝视")

        self.assertEqual(parse_shot_numbers("3-5", 10), (False, [3, 4, 5]))
        self.assertEqual(parse_shot_numbers("6,9", 10), (False, [6, 9]))
        self.assertEqual(parse_shot_numbers("0", 4), (True, [1, 2, 3, 4]))

    def test_vault_store_reserve_short_shot_64bit_and_lineage_dag(self):
        self.assertEqual(sanitize_project_name("../../bad:name?"), "bad_name")
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultStore(os.path.join(tmp, "proj"))
            t1, lock1 = store.reserve_next_take(1)
            t2, lock2 = store.reserve_next_take(1)
            self.assertEqual(t1, "shot_0001_take_001")
            self.assertEqual(t2, "shot_0001_take_002")
            store.release_reserve(lock1)
            store.release_reserve(lock2)

            p_shot1 = torch.rand((40, 64, 64, 3), dtype=torch.float32)
            p_hash1 = sha256_text("shot1")
            seed64 = "18446744073709551615"
            store.save_take_context(
                take_id=t1, shot_index=1, clean_images=p_shot1,
                audio_waveform_48k=torch.zeros((2, 48000)), audio_sr=48000,
                fps=24, prompt_sha256=p_hash1, parent_take_id="", seed_text=seed64,
                keep_frames=40, trim_head_frames=0, shot_type=1,
            )
            ctx1 = store.load_take_context(t1)
            self.assertEqual(ctx1["meta"]["seed_text"], seed64)
            self.assertEqual(ctx1["tensors"]["clean_tail"].dtype, torch.float16)

            short_img = torch.rand((14, 64, 64, 3), dtype=torch.float32)
            t2_s, lk = store.reserve_next_take(2)
            store.release_reserve(lk)
            store.save_take_context(
                take_id=t2_s, shot_index=2, clean_images=short_img,
                audio_waveform_48k=None, audio_sr=48000, fps=24,
                prompt_sha256=sha256_text("shot2"), parent_take_id=t1, seed_text="42",
                keep_frames=14, trim_head_frames=22, shot_type=1,
                parent_clean_tail=ctx1["tensors"]["clean_tail"], max_clean_tail_frames=39,
            )
            ctx2 = store.load_take_context(t2_s)
            self.assertEqual(ctx2["tensors"]["clean_tail"].shape[0], 39)

            t3_s, lk3 = store.reserve_next_take(3)
            store.release_reserve(lk3)
            store.save_take_context(
                take_id=t3_s, shot_index=3, clean_images=p_shot1,
                audio_waveform_48k=None, audio_sr=48000, fps=24,
                prompt_sha256=sha256_text("shot3"), parent_take_id="", seed_text="99",
                keep_frames=40, trim_head_frames=0, shot_type=1,
            )

            plan = [
                {"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": p_hash1},
                {"shot_index": 2, "shot_id": "shot_0002", "is_continuation": True, "prompt_sha256": sha256_text("shot2")},
                {"shot_index": 3, "shot_id": "shot_0003", "is_continuation": False, "prompt_sha256": sha256_text("shot3")},
            ]
            self.assertEqual([d["status"] for d in LineageDAG.evaluate(plan, store.load_project())], ["VALID", "VALID", "VALID"])

            plan_edited = [dict(x) for x in plan]
            plan_edited[1]["prompt_sha256"] = sha256_text("shot2_modified")
            dag_ed = LineageDAG.evaluate(plan_edited, store.load_project())
            self.assertEqual(dag_ed[1]["status"], "VALID")
            self.assertTrue(dag_ed[1]["prompt_edited"])

            t1_new, lkn = store.reserve_next_take(1)
            store.release_reserve(lkn)
            store.save_take_context(
                take_id=t1_new, shot_index=1, clean_images=p_shot1,
                audio_waveform_48k=None, audio_sr=48000, fps=24,
                prompt_sha256=p_hash1, parent_take_id="", seed_text="100",
                keep_frames=40, trim_head_frames=0, shot_type=1,
            )
            self.assertEqual([d["status"] for d in LineageDAG.evaluate(plan, store.load_project())], ["VALID", "STALE", "VALID"])

    def test_net_preserve_duration_mode_schedule(self):
        sched = compute_cumulative_schedule(
            [2.0, 2.0, 2.5], fps=24, is_continuations=[False, True, True],
            context_frames=22, duration_mode="net_preserve",
        )
        self.assertEqual([s["keep_frames"] for s in sched], [48, 48, 60])
        self.assertEqual([s["trim_head_frames"] for s in sched], [0, 22, 22])
        for s in sched:
            self.assertEqual((s["target_frames"] - 5) % 17, 0)

    def test_cinema_pack_json_parser_and_directives(self):
        cpack = (
            '{"shots": ['
            '{"id": 1, "duration": 2.5, "shot_type": 0, "hard_cut": true, "prompt": "静默凝视", "audio_file": "s1.wav"},'
            '{"id": 2, "duration": 3.0, "shot_type": 1, "segment_only": true, "prompt": "长拳破空"}'
            ']}'
        )
        shots = parse_script(cpack, common_prompt="冷色调", script_format="cinema_pack_json")
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["shot_type"], 0)
        self.assertTrue(shots[0]["is_drama_mode"])
        self.assertFalse(shots[0]["is_continuation"])
        self.assertEqual(shots[0]["audio_file"], "s1.wav")
        self.assertTrue(shots[1]["segment_only"])
        self.assertIn("冷色调", shots[1]["prompt"])

        md = "& 2.0 &\n@cut\n@segment\n@audio_file \"lines/a1.wav\"\n出拳"
        md_shots = parse_script(md, script_format="1114_markdown", has_seed_video=True)
        self.assertFalse(md_shots[0]["is_continuation"])
        self.assertTrue(md_shots[0]["segment_only"])
        self.assertEqual(md_shots[0]["audio_file"], "lines/a1.wav")

    def test_vault_store_cleanup_unused_takes_to_recycle_bin(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            store = VaultStore(os.path.join(tmp, "RecycleProj"))
            img = torch.rand((24, 32, 32, 3), dtype=torch.float32)
            h = sha256_text("p1")
            t1_old, lk1 = store.reserve_next_take(1)
            store.release_reserve(lk1)
            store.save_take_context(t1_old, 1, img, None, 48000, 24, h, "", "1", 24, 0, 1)
            t1_act, lk2 = store.reserve_next_take(1)
            store.release_reserve(lk2)
            store.save_take_context(t1_act, 1, img, None, 48000, 24, h, "", "2", 24, 0, 1, set_active=True)

            with patch.dict(sys.modules, {"send2trash": None}):
                rep = store.cleanup_unused_takes()
            self.assertEqual(rep["recycled_count"], 1)
            self.assertIn(f"{t1_old}.h3ctx.safetensors", rep["recycled_files"])
            self.assertFalse(os.path.isfile(os.path.join(store.vault_dir, f"{t1_old}.h3ctx.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(store.vault_dir, f"{t1_act}.h3ctx.safetensors")))
            self.assertTrue(any(t1_old in fn for fn in os.listdir(store.recycle_dir)))

    def test_vault_store_overhang_roundtrip_exact(self):
        import comfy.nested_tensor  # type: ignore

        with tempfile.TemporaryDirectory() as tmp:
            store = VaultStore(os.path.join(tmp, "OverhangProj"))
            tid, lk = store.reserve_next_take(1)
            store.release_reserve(lk)
            lat = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 17, 2, 2)), torch.zeros((1, 32, 2, 93))))}
            meta = store.save_take_context(
                tid, 1, torch.rand((56, 32, 32, 3)), None, 48000, 24,
                sha256_text("ov"), "", "1", 56, 0, 1, sampled_latent=lat, raw_trimmed_frames=56,
            )
            self.assertTrue(meta["has_tail_latent"])
            ctx = store.load_take_context(tid)
            self.assertAlmostEqual(ctx["meta"]["audio_overhang"], -1.0 / 3.0, places=4)
            self.assertEqual(resolve_audio_guide_end(22, ctx["meta"]["audio_overhang"]), 36)

    def test_lineage_dag_hard_cut_blocks_upstream_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultStore(os.path.join(tmp, "CutBlockProj"))
            img = torch.rand((24, 32, 32, 3))
            hashes = [sha256_text(f"s{i}") for i in range(1, 5)]
            t1, l1 = store.reserve_next_take(1); store.release_reserve(l1)
            store.save_take_context(t1, 1, img, None, 48000, 24, hashes[0], "", "1", 24, 0, 1)
            t2, l2 = store.reserve_next_take(2); store.release_reserve(l2)
            store.save_take_context(t2, 2, img, None, 48000, 24, hashes[1], t1, "2", 24, 22, 1)
            t3, l3 = store.reserve_next_take(3); store.release_reserve(l3)
            store.save_take_context(t3, 3, img, None, 48000, 24, hashes[2], "", "3", 24, 0, 1)
            t4, l4 = store.reserve_next_take(4); store.release_reserve(l4)
            store.save_take_context(t4, 4, img, None, 48000, 24, hashes[3], t3, "4", 24, 22, 1)

            t1b, l1b = store.reserve_next_take(1); store.release_reserve(l1b)
            store.save_take_context(t1b, 1, img, None, 48000, 24, hashes[0], "", "9", 24, 0, 1)
            plan = [
                {"shot_index": 1, "shot_id": "shot_0001", "is_continuation": False, "prompt_sha256": hashes[0]},
                {"shot_index": 2, "shot_id": "shot_0002", "is_continuation": True, "prompt_sha256": hashes[1]},
                {"shot_index": 3, "shot_id": "shot_0003", "is_continuation": False, "prompt_sha256": hashes[2]},
                {"shot_index": 4, "shot_id": "shot_0004", "is_continuation": True, "prompt_sha256": hashes[3]},
            ]
            self.assertEqual([d["status"] for d in LineageDAG.evaluate(plan, store.load_project())], ["VALID", "STALE", "VALID", "VALID"])

    def test_shot_numbers_clamped_and_deduplicated(self):
        self.assertEqual(parse_shot_numbers("2-99", 4), (False, [2, 3, 4]))
        self.assertEqual(parse_shot_numbers("3,1,3,99", 4), (False, [3, 1]))
        self.assertEqual(parse_shot_numbers("bad", 4), (False, [1]))

    def test_aspect_presets_and_canvas_16px_alignment(self):
        for preset in grid_math.ASPECT_PRESETS:
            for mp in (0.25, 1.0, 1.5):
                w, h = grid_math.adapt_canvas_from_ratio(preset, megapixels=mp)
                self.assertEqual(w % 32, 0)
                self.assertEqual(h % 32, 0)
                self.assertEqual(w % 16, 0)
                self.assertEqual(h % 16, 0)
                self.assertGreaterEqual(w, 64)
                self.assertGreaterEqual(h, 64)


if __name__ == "__main__":
    unittest.main()
