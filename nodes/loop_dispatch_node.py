"""[N2D] ShotVaultLoopDispatch: Stateless external-loop segment & 9-ref dispatcher (no internal loop)."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..compat import resample_audio, sha256_text
from ..core.asset_and_turnaround import build_continuation_driving_audio, load_shot_multimodal_refs, slice_1114_prev_audio_guide
from ..core.grid_math import AUDIO_VAE_ALIGN, AUDIO_VAE_SR, audio_latent_t, compute_cumulative_schedule, compute_tail_bridge_plan, resolve_duration_mode
from ..core.vault_store import VaultStore, compute_continuation_depth, load_next_shot_head_ref, load_scene_anchor_clean_ref
from ..server_routes import _get_input_root


def _resolve_dispatch_seed(
    seed_mode: str,
    base_seed: int,
    seed_text: str,
    curr_idx: int,
    shot: dict[str, Any],
    retry_lock: dict[str, Any],
    honor_retry: bool,
    active_meta: Optional[dict] = None,
) -> tuple[int, str]:
    if honor_retry and retry_lock.get("enabled"):
        rm = str(retry_lock.get("seed_mode") or "randomize").lower()
        rtxt = str(retry_lock.get("seed_text") or "").strip()
        prev_s = str((active_meta or {}).get("seed_text") or "").strip()
        if rm == "randomize":
            val = random.SystemRandom().randint(1, 0xFFFFFFFFFFFFFFFF)
            return val, str(val)
        if rm == "delta_+1":
            try:
                bval = int(rtxt, 0) if rtxt else (int(prev_s) if prev_s.isdigit() else int(base_seed))
            except ValueError:
                bval = int(prev_s) if prev_s.isdigit() else int(base_seed)
            val = (bval + 1) & 0xFFFFFFFFFFFFFFFF
            return val, str(val)
        if rm in ("fixed", "custom", "custom_string", "original"):
            if rtxt:
                try:
                    val = int(rtxt, 0) & 0xFFFFFFFFFFFFFFFF
                    return val, str(val)
                except ValueError:
                    pass
            if prev_s.isdigit():
                val = int(prev_s) & 0xFFFFFFFFFFFFFFFF
                return val, str(val)
            val = int(base_seed) & 0xFFFFFFFFFFFFFFFF
            return val, str(val)
    if seed_mode == "randomize":
        val = random.SystemRandom().randint(1, 0xFFFFFFFFFFFFFFFF)
        return val, str(val)
    if seed_mode == "custom_string" and (seed_text or "").strip():
        try:
            val = int(seed_text.strip(), 0) & 0xFFFFFFFFFFFFFFFF
            return val, str(val)
        except ValueError:
            pass
    if seed_mode == "fixed":
        val = int(base_seed) & 0xFFFFFFFFFFFFFFFF
        return val, str(val)
    shot_seed_txt = str(shot.get("seed_text") or "").strip()
    if shot_seed_txt:
        try:
            val = int(shot_seed_txt, 0) & 0xFFFFFFFFFFFFFFFF
            return val, str(val)
        except ValueError:
            pass
    val = (int(base_seed) + int(curr_idx) - 1) & 0xFFFFFFFFFFFFFFFF
    return val, str(val)


class ShotVaultLoopDispatch:
    """[N2D] Loop-agnostic per-segment dispatcher outputting gear params and 9 ref images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_pipe": ("SHOT_PIPE",),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "index_base": (["0_based (easy_forLoop)", "1_based (1..N)"], {"default": "0_based (easy_forLoop)"}),
                "honor_director_retry_lock": ("BOOLEAN", {"default": True}),
                "seed_mode": (["from_plan_or_increment", "randomize", "fixed", "custom_string"], {"default": "from_plan_or_increment"}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_text": ("STRING", {"default": ""}),
            },
            "optional": {
                "loop_carry": ("*",),
            },
        }

    RETURN_TYPES = (
        "SHOT_PIPE", "INT", "INT", "STRING", "INT", "INT", "INT", "INT",
        "BOOLEAN", "BOOLEAN", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "shot_pipe", "total_segments", "current_segment", "prompt", "width", "height", "target_frames", "keep_frames",
        "is_drama_mode", "is_continuation", "overlap_frames", "trim_head_frames", "taper_tail_frames", "taper_ramp_frames",
        "audio_guide_frames", "audio_feather_ticks", "seed",
        "ref_image_1", "ref_image_2", "ref_image_3", "ref_image_4", "ref_image_5",
        "ref_image_6", "ref_image_7", "ref_image_8", "ref_image_9",
        "prev_clean_images",
    )
    FUNCTION = "dispatch"
    CATEGORY = "ShotVault/Core"

    @classmethod
    def IS_CHANGED(
        cls,
        shot_pipe: dict,
        segment_index: int = 0,
        index_base: str = "0_based (easy_forLoop)",
        honor_director_retry_lock: bool = True,
        seed_mode: str = "from_plan_or_increment",
        base_seed: int = 42,
        seed_text: str = "",
        loop_carry: Any = None,
    ):
        retry_lock = (shot_pipe or {}).get("retry_lock") or {}
        if seed_mode == "randomize":
            return float("nan")
        if honor_retry := bool(honor_director_retry_lock and retry_lock.get("enabled")):
            if str(retry_lock.get("seed_mode") or "randomize").lower() == "randomize":
                return float("nan")
        shots = (shot_pipe or {}).get("shots") or []
        total = max(1, len(shots))
        raw_i = (int(segment_index) + 1) if str(index_base).startswith("0_based") else int(segment_index)
        curr_idx = int(retry_lock.get("shot_index") or raw_i) if honor_retry else raw_i
        curr_idx = max(1, min(total, curr_idx))
        shot = shots[curr_idx - 1] if 1 <= curr_idx <= len(shots) else {}
        project_root = str((shot_pipe or {}).get("project_root", ""))
        parent_sig = ""
        if project_root:
            st = VaultStore(project_root)
            parent_sig = st.get_active_take_id(curr_idx - 1) if curr_idx > 1 else st.get_active_take_id(0)
            curr_sig = st.get_active_take_id(curr_idx)
            next_sig = st.get_active_take_id(curr_idx + 1) if (honor_retry and int(retry_lock.get("tail_bridge_frames") or 0) in (1, 5, 22)) else ""
        else:
            curr_sig = ""
            next_sig = ""
        audio_sig = str(((shot_pipe or {}).get("per_shot_audio") or {}).get(curr_idx, {}).get("sha256", ""))
        carry_sig = str(loop_carry.get("last_saved_take_id", "")) if isinstance(loop_carry, dict) else str(loop_carry is not None)
        key = "|".join([
            project_root,
            str(curr_idx),
            str((shot_pipe or {}).get("context_frames", 22)),
            str((shot_pipe or {}).get("audio_drive_active", False)),
            str(shot.get("overlap_override", 0)),
            str(shot.get("prompt_sha256", "")),
            str(shot.get("ref_slots", shot.get("ref_images", []))),
            str(retry_lock if honor_retry else ""),
            str((shot_pipe or {}).get("anti_degradation", "")),
            parent_sig, curr_sig, next_sig, audio_sig, carry_sig,
            str(seed_mode), str(base_seed), str(seed_text),
        ])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def dispatch(
        self,
        shot_pipe: dict[str, Any],
        segment_index: int = 0,
        index_base: str = "0_based (easy_forLoop)",
        honor_director_retry_lock: bool = True,
        seed_mode: str = "from_plan_or_increment",
        base_seed: int = 42,
        seed_text: str = "",
        loop_carry: Any = None,
    ):
        shots = list(shot_pipe.get("shots") or ())
        total = max(1, len(shots))
        fps = int(shot_pipe.get("fps", 24))
        width = int(shot_pipe.get("width", 1344))
        height = int(shot_pipe.get("height", 768))
        project_root = str(shot_pipe.get("project_root") or "")
        retry_lock = dict(shot_pipe.get("retry_lock") or {})
        use_retry = bool(honor_director_retry_lock and retry_lock.get("enabled"))

        raw_i = (int(segment_index) + 1) if str(index_base).startswith("0_based") else int(segment_index)
        curr_idx = max(1, min(total, int(retry_lock.get("shot_index") or raw_i) if use_retry else raw_i))

        audio_drive_active = bool(shot_pipe.get("audio_drive_active"))
        effective_ctx = 39 if audio_drive_active else int(shot_pipe.get("context_frames", 22))
        overlap_overrides = (
            [39 if s["is_continuation"] else 0 for s in shots]
            if audio_drive_active
            else ([int(s.get("overlap_override") or 0) for s in shots] or [0])
        )
        schedule = compute_cumulative_schedule(
            [s["duration_sec"] for s in shots] or [5.0],
            fps=fps,
            is_continuations=[s["is_continuation"] for s in shots] or [False],
            context_frames=effective_ctx,
            duration_mode=resolve_duration_mode(str(shot_pipe.get("duration_mode", "cumulative_cursor")), audio_drive_active),
            overlap_overrides=overlap_overrides,
        )
        sched = schedule[curr_idx - 1]
        shot = dict(shots[curr_idx - 1]) if shots else {
            "shot_index": 1, "shot_id": "shot_0001", "duration_sec": 5.0,
            "shot_type": 1, "is_drama_mode": False, "is_continuation": False,
            "prompt": "", "local_prompt": "", "prompt_sha256": sha256_text(""),
            "speech_spans": [], "inherit_global_refs": True,
        }
        prompt = str(shot.get("prompt") or "")
        if use_retry and str(retry_lock.get("prompt_override") or "").strip():
            po = str(retry_lock["prompt_override"]).strip()
            common = str(shot_pipe.get("common_prompt") or "").strip()
            prompt = f"{common}\n{po}" if (common and not po.startswith(common)) else po
            shot["local_prompt"] = po
            shot["prompt"] = prompt
            shot["prompt_sha256"] = sha256_text(prompt)

        store = VaultStore(project_root) if project_root else None
        is_cont = bool(sched["is_continuation"])
        parent_tid = ""
        parent_ctx = None
        seed_ctx = None
        active_meta = None
        if store is not None:
            proj = store.load_project()
            cur_rec = (proj.get("shots") or {}).get(str(curr_idx)) or {}
            old_active_tid = str(cur_rec.get("active_take_id") or "")
            active_meta = (cur_rec.get("takes") or {}).get(old_active_tid)
            if is_cont:
                parent_tid = store.get_active_take_id(curr_idx - 1) if curr_idx > 1 else store.get_active_take_id(0)
                if parent_tid:
                    parent_ctx = store.load_take_context(parent_tid)
                if curr_idx == 1:
                    seed_ctx = store.load_take_context("shot_0000_seed")
                    if parent_ctx is None and seed_ctx is not None:
                        parent_ctx = seed_ctx
                        parent_tid = "shot_0000_seed"
        if parent_ctx is None:
            is_cont = False

        prev_clean_images = torch.zeros((4, 64, 64, 3), dtype=torch.float32)
        prev_audio = {"waveform": torch.zeros((1, 2, 1600), dtype=torch.float32), "sample_rate": AUDIO_VAE_SR}
        prev_tail_latent = None
        if is_cont and parent_ctx is not None:
            tensors = parent_ctx["tensors"]
            ov = max(1, int(sched["overlap_frames"] or 22))
            ag = int(sched.get("audio_guide_frames") or max(1, ov - 2))
            if "clean_tail" in tensors:
                prev_clean_images = tensors["clean_tail"][-ov:].float().contiguous()
            if "audio_32k_tail" in tensors:
                prev_audio = slice_1114_prev_audio_guide(tensors["audio_32k_tail"], ov, ag, fps=fps)
            if "tail_latent_video" in tensors:
                try:
                    import comfy.nested_tensor  # type: ignore

                    v_lat = tensors["tail_latent_video"]
                    a_lat = tensors.get("tail_latent_audio")
                    samples = comfy.nested_tensor.NestedTensor((v_lat, a_lat)) if a_lat is not None else v_lat
                    prev_tail_latent = {
                        "samples": samples,
                        "shotvault_meta": {"audio_overhang": float(parent_ctx["meta"].get("audio_overhang", 0.0))},
                    }
                except Exception:
                    prev_tail_latent = None

        resolved_seed, resolved_seed_text = _resolve_dispatch_seed(
            seed_mode, base_seed, seed_text, curr_idx, shot, retry_lock, bool(honor_director_retry_lock), active_meta=active_meta,
        )

        mm_refs = load_shot_multimodal_refs(
            shot,
            global_cfg=shot_pipe.get("global_refs_cfg") or {},
            input_dir=_get_input_root(),
            project_root=project_root,
            auto_split_turnaround=bool(shot_pipe.get("auto_split_turnaround", False)),
            fps=fps,
            max_frames=362,
            skip_ref_audios=audio_drive_active,
        )
        imgs = list(mm_refs["images"])[:9]
        slot_imgs = list(mm_refs.get("slot_images") or [])
        padded_imgs: list[Optional[torch.Tensor]] = (slot_imgs + [None] * 9)[:9] if slot_imgs else (imgs + [None] * (9 - len(imgs)))

        per_shot_audio_map = shot_pipe.get("per_shot_audio") or {}
        shot_audio_ref = per_shot_audio_map.get(curr_idx)
        shot_audio_guide, delivery_ref_audio = build_continuation_driving_audio(
            shot_audio_ref,
            curr_idx=curr_idx,
            sched=sched,
            fps=fps,
            av_sync_lead_sec=0.0,
            parent_ctx=parent_ctx,
            per_shot_audio_map=per_shot_audio_map,
            seed_ctx=seed_ctx,
        )

        # Bidirectional Bridge Reroll: optionally load clean delivered head of next shot (`curr_idx + 1`)
        tb_frames = 0
        tb_index = -1
        tb_noise = str(retry_lock.get("tail_bridge_noise") or "clean")
        next_head_clean_ref = None
        next_head_audio_32k = None
        bridged_next_take_id = ""
        req_tb = int(retry_lock.get("tail_bridge_frames") or 0) if use_retry else 0
        if use_retry and req_tb in (1, 5, 22) and store is not None and curr_idx < total:
            next_ref = load_next_shot_head_ref(store, curr_idx + 1, req_tb, fps=fps)
            if next_ref is not None:
                tb_plan = compute_tail_bridge_plan(
                    sched["trim_head_frames"],
                    sched["keep_frames"],
                    next_ref["head_frames"],
                    audio_drive_active=audio_drive_active,
                )
                if tb_plan["tail_bridge_frames"] in (1, 5, 22):
                    sched = dict(sched)
                    sched["target_frames"] = int(tb_plan["target_frames"])
                    sched["keep_frames"] = int(tb_plan["keep_frames"])
                    tb_frames = int(tb_plan["tail_bridge_frames"])
                    tb_index = int(tb_plan["tail_anchor_index"])
                    next_head_clean_ref = next_ref["images"]
                    next_head_audio_32k = next_ref.get("audio_32k")
                    bridged_next_take_id = str(next_ref.get("next_take_id") or "")

        trim_head = int(sched["trim_head_frames"])
        shifted_speech = [(int(s) + trim_head, int(e) + trim_head) for s, e in (shot.get("speech_spans") or ()) if int(e) > int(s)]

        cont_depth = compute_continuation_depth(
            shots, curr_idx, is_cont, has_seed=bool(store is not None and store.get_active_take_id(0))
        )
        active_pipe = dict(shot_pipe)
        active_pipe.update({
            "current_shot_index": curr_idx,
            "current_shot": shot,
            "parent_take_id": parent_tid,
            "is_continuation": is_cont,
            "continuation_depth": cont_depth,
            "tail_bridge_frames": tb_frames,
            "tail_bridge_index": tb_index,
            "tail_bridge_noise": tb_noise,
            "next_head_clean_ref": next_head_clean_ref,
            "next_head_audio_32k": next_head_audio_32k,
            "bridged_next_take_id": bridged_next_take_id,
            "target_frames": int(sched["target_frames"]),
            "trim_head_frames": trim_head,
            "keep_frames": int(sched["keep_frames"]),
            "overlap_frames": int(sched["overlap_frames"]),
            "overlap_gear": int(sched["overlap_gear"]),
            "taper_tail_frames": int(sched["taper_tail_frames"]),
            "taper_ramp_frames": int(sched["taper_ramp_frames"]),
            "audio_guide_frames": int(sched["audio_guide_frames"]),
            "gear_feather_ticks": int(sched["audio_feather_ticks"]),
            "auto_scale_feather": True,
            "speech_spans": shifted_speech,
            "seed_text": resolved_seed_text,
            "prompt": prompt,
            "prompt_sha256": shot.get("prompt_sha256", sha256_text(prompt)),
            "shot_type": int(shot.get("shot_type", 1)),
            "delivery_ref_audio": delivery_ref_audio,
            "prev_audio": prev_audio if is_cont else {"waveform": torch.zeros((1, 2, 1600), dtype=torch.float32), "sample_rate": AUDIO_VAE_SR},
            "prev_tail_latent": prev_tail_latent,
            "parent_clean_tail_ref": prev_clean_images if is_cont else None,
            "scene_anchor_clean_ref": load_scene_anchor_clean_ref(store, shots, curr_idx, is_cont, parent_tid, prev_clean_images) if is_cont else None,
            "parent_audio_32k_ref": prev_audio if is_cont else None,
            "shot_audio_guide": shot_audio_guide,
            "loaded_ref_images": imgs,
            "slot_to_ordinal": mm_refs.get("slot_to_ordinal") or {},
            "loaded_ref_videos": mm_refs["videos"],
            "loaded_ref_audios": mm_refs["audios"],
            "active_ref_summary": mm_refs["summary"],
        })

        return (
            active_pipe,
            total,
            curr_idx,
            prompt,
            width,
            height,
            int(sched["target_frames"]),
            int(sched["keep_frames"]),
            bool(shot.get("is_drama_mode", False)),
            bool(sched["is_continuation"]),
            int(sched["overlap_frames"]),
            trim_head,
            int(sched["taper_tail_frames"]),
            int(sched["taper_ramp_frames"]),
            int(sched["audio_guide_frames"]),
            int(sched["audio_feather_ticks"]),
            resolved_seed,
            *padded_imgs,
            prev_clean_images,
        )
