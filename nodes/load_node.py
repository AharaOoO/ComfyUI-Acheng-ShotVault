"""[N2] ShotVaultLoad: KJ-restrained 10-port loader, end-aligned 800-sample audio, turnaround assets & DAG resolver."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..compat import resample_audio, sha256_text
from ..core.asset_and_turnaround import build_continuation_driving_audio, load_shot_multimodal_refs, slice_1114_prev_audio_guide
from ..core.grid_math import (
    AUDIO_VAE_ALIGN,
    AUDIO_VAE_SR,
    audio_latent_t,
    compute_cumulative_schedule,
    pad_audio_800,
    resolve_duration_mode,
)
from ..core.lineage_dag import LineageDAG
from ..core.script_parser import parse_shot_numbers
from ..core.seed_video_ingest import ingest_seed_video
from ..core.vault_store import VaultStore, compute_continuation_depth, load_scene_anchor_clean_ref
from ..server_routes import _get_input_root, register_current_running_prompt


def _dummy_images() -> torch.Tensor:
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _dummy_audio() -> dict[str, Any]:
    return {"waveform": torch.zeros((1, 2, 1600), dtype=torch.float32), "sample_rate": AUDIO_VAE_SR}


def _resolve_seed(seed_mode: str, shot_seed: int, seed_text: str, prev_take_meta: Optional[dict] = None) -> tuple[int, str]:
    txt = (seed_text or "").strip()
    if seed_mode == "custom_string" and txt:
        try:
            val = int(txt, 0) & 0xFFFFFFFFFFFFFFFF
            return val, str(val)
        except ValueError:
            pass
    if seed_mode == "original_retry" and prev_take_meta and str(prev_take_meta.get("seed_text", "")).isdigit():
        val = int(prev_take_meta["seed_text"]) & 0xFFFFFFFFFFFFFFFF
        return val, str(val)
    if seed_mode == "randomize":
        val = random.SystemRandom().randint(1, 0xFFFFFFFFFFFFFFFF)
        return val, str(val)
    if seed_mode == "increment":
        val = (int(shot_seed) + 1) & 0xFFFFFFFFFFFFFFFF
        return val, str(val)
    val = int(shot_seed) & 0xFFFFFFFFFFFFFFFF
    return val, str(val)


class ShotVaultLoad:
    """[N2] Compact 10-output loader: timing, clean tail, drama switch, prompt, canvas, seed & split assets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_pipe": ("SHOT_PIPE",),
                "start_mode": (["sequence_auto", "manual_picks"], {"default": "sequence_auto"}),
                "shot_numbers": ("STRING", {"default": "1"}),
                "continue_from": (["auto_latent_first", "pixel_clean_taper", "hard_cut"], {"default": "auto_latent_first"}),
                "seed_mode": (["randomize", "fixed", "increment", "original_retry", "custom_string"], {"default": "randomize"}),
                "shot_seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_text": ("STRING", {"default": ""}),
                "av_sync_lead_sec": ("FLOAT", {"default": 0.00, "min": -0.50, "max": 1.50, "step": 0.04}),
            },
            "optional": {
                "parent_take_override": ("STRING", {"default": ""}),
                "keep_downstream_on_reroll": ("BOOLEAN", {"default": False}),
                "reuse_seed_images": ("IMAGE",),
                "reuse_seed_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (
        "SHOT_PIPE", "IMAGE", "BOOLEAN", "STRING", "INT", "INT", "INT", "INT", "IMAGE", "IMAGE",
    )
    RETURN_NAMES = (
        "shot_pipe", "prev_clean_images", "is_drama_mode", "prompt", "width", "height", "target_frames", "seed", "asset_ref_1", "asset_ref_2",
    )
    FUNCTION = "load_shot"
    CATEGORY = "ShotVault/Core"

    @classmethod
    def IS_CHANGED(
        cls,
        shot_pipe: dict,
        start_mode: str,
        shot_numbers: str,
        continue_from: str,
        seed_mode: str,
        shot_seed: int = 42,
        seed_text: str = "",
        av_sync_lead_sec: float = 0.0,
        parent_take_override: str = "",
        keep_downstream_on_reroll: bool = False,
        reuse_seed_images=None,
        **kwargs,
    ):
        if "seed" in kwargs:
            shot_seed = int(kwargs["seed"])
        retry_lock = (shot_pipe or {}).get("retry_lock") or {}
        if retry_lock.get("enabled") and str(retry_lock.get("seed_mode") or "randomize").lower() == "randomize":
            return float("nan")
        if not retry_lock.get("enabled") and seed_mode == "randomize":
            return float("nan")
        project_root = str((shot_pipe or {}).get("project_root") or "")
        store = VaultStore(project_root) if project_root else None
        shots = (shot_pipe or {}).get("shots") or []
        if retry_lock.get("enabled") and shots:
            curr_idx = max(1, min(len(shots), int(retry_lock.get("shot_index") or 1)))
        else:
            curr_idx = cls._pick_shot_index(shots, store, start_mode, shot_numbers)[0]
        shot = shots[curr_idx - 1] if 1 <= curr_idx <= len(shots) else {}
        parent_tid = (parent_take_override or "").strip()
        if not parent_tid and store is not None and curr_idx > 1:
            parent_tid = store.get_active_take_id(curr_idx - 1)
        elif not parent_tid and store is not None and curr_idx == 1:
            parent_tid = store.get_active_take_id(0)
        audio_sig = str(((shot_pipe or {}).get("per_shot_audio") or {}).get(curr_idx, {}).get("sha256", ""))
        key = "|".join([
            project_root, str(curr_idx), str(shot.get("prompt_sha256", "")),
            str(shot_seed), str(seed_text), seed_mode, continue_from,
            f"{av_sync_lead_sec:.3f}", parent_tid, audio_sig, str(retry_lock),
            str((shot_pipe or {}).get("audio_drive_active", False)),
            str((shot_pipe or {}).get("context_frames", 22)),
        ])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _pick_shot_index(shots: list[dict], store: Optional[VaultStore], start_mode: str, shot_numbers: str) -> tuple[int, list[int]]:
        total = max(1, len(shots))
        is_reset, picks = parse_shot_numbers(shot_numbers, total)
        if is_reset:
            return 1, list(range(2, total + 1))
        if start_mode == "manual_picks":
            return picks[0], picks[1:]
        if len(picks) > 1 or picks[0] > 1:
            return picks[0], picks[1:]
        if store is not None and shots:
            dag = LineageDAG.evaluate(shots, store.load_project())
            pending = [d["shot_index"] for d in dag if d["status"] in ("UNRENDERED", "STALE")]
            if pending:
                return pending[0], pending[1:]
        return picks[0], list(range(picks[0] + 1, total + 1))

    def load_shot(
        self,
        shot_pipe: dict,
        start_mode: str,
        shot_numbers: str,
        continue_from: str,
        seed_mode: str,
        shot_seed: int = 42,
        seed_text: str = "",
        av_sync_lead_sec: float = 0.0,
        parent_take_override: str = "",
        keep_downstream_on_reroll: bool = False,
        reuse_seed_images: Optional[torch.Tensor] = None,
        reuse_seed_audio: Optional[dict] = None,
        **kwargs,
    ):
        if "seed" in kwargs:
            shot_seed = int(kwargs["seed"])
        register_current_running_prompt()

        project_root = str(shot_pipe["project_root"])
        store = VaultStore(project_root)
        fps = int(shot_pipe.get("fps", 24))
        width = int(shot_pipe.get("width", 1344))
        height = int(shot_pipe.get("height", 768))
        shots = list(shot_pipe.get("shots") or ())

        if reuse_seed_images is not None and isinstance(reuse_seed_images, torch.Tensor) and reuse_seed_images.shape[0] > 0:
            ingest_seed_video(
                store,
                width=width,
                height=height,
                fps=fps,
                reuse_seed_images=reuse_seed_images,
                reuse_seed_audio=reuse_seed_audio,
            )
            if shots:
                shots[0]["is_continuation"] = True

        retry_lock = dict(shot_pipe.get("retry_lock") or {})
        use_retry = bool(retry_lock.get("enabled") and shots)
        if use_retry:
            curr_idx = max(1, min(len(shots), int(retry_lock.get("shot_index") or 1)))
            remaining = []
        else:
            curr_idx, remaining = self._pick_shot_index(shots, store, start_mode, shot_numbers)

        audio_drive_active = bool(shot_pipe.get("audio_drive_active"))
        effective_ctx = 39 if audio_drive_active else int(shot_pipe.get("context_frames", 22))
        overlap_overrides = (
            [39 if s["is_continuation"] else 0 for s in shots]
            if audio_drive_active
            else [int(s.get("overlap_override") or 0) for s in shots]
        )
        schedule = compute_cumulative_schedule(
            [s["duration_sec"] for s in shots],
            fps=fps,
            is_continuations=[s["is_continuation"] for s in shots],
            context_frames=effective_ctx,
            av_sync_lead_sec=float(av_sync_lead_sec),
            duration_mode=resolve_duration_mode(str(shot_pipe.get("duration_mode", "cumulative_cursor")), audio_drive_active),
            overlap_overrides=overlap_overrides,
        )
        shot = dict(shots[curr_idx - 1])
        if use_retry:
            po = str(retry_lock.get("prompt_override") or "").strip()
            if po:
                common = str(shot_pipe.get("common_prompt") or "").strip()
                shot["local_prompt"] = po
                shot["prompt"] = f"{common}\n{po}" if (common and not po.startswith(common)) else po
                shot["prompt_sha256"] = sha256_text(shot["prompt"])
        sched = schedule[curr_idx - 1]
        is_cont = bool(sched["is_continuation"]) and (continue_from != "hard_cut")

        parent_tid = ""
        parent_ctx = None
        if is_cont:
            parent_tid = (parent_take_override or "").strip() or store.get_active_take_id(curr_idx - 1)
            if not parent_tid and curr_idx == 1:
                parent_tid = store.get_active_take_id(0)
            if parent_tid:
                parent_ctx = store.load_take_context(parent_tid)
        if parent_ctx is None:
            is_cont = False
            if sched["is_continuation"] and continue_from != "hard_cut" and not parent_tid:
                sched = dict(sched, trim_head_frames=max(0, round(float(av_sync_lead_sec) * fps)), overlap_frames=0)

        prev_clean_images = _dummy_images()
        prev_audio = _dummy_audio()
        prev_tail_latent = None
        if is_cont and parent_ctx is not None:
            tensors = parent_ctx["tensors"]
            ov = max(1, int(sched["overlap_frames"] or 22))
            ag = int(sched.get("audio_guide_frames") or max(1, ov - 2))
            if "clean_tail" in tensors:
                prev_clean_images = tensors["clean_tail"][-ov:].float().contiguous()
            if "audio_32k_tail" in tensors:
                prev_audio = slice_1114_prev_audio_guide(tensors["audio_32k_tail"], ov, ag, fps=fps)
            if continue_from == "auto_latent_first" and "tail_latent_video" in tensors:
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

        per_shot_audio_map = shot_pipe.get("per_shot_audio") or {}
        shot_audio_ref = per_shot_audio_map.get(curr_idx)
        seed_ctx = store.load_take_context("shot_0000_seed") if (curr_idx == 1 and is_cont) else None
        shot_audio_guide, delivery_ref_audio = build_continuation_driving_audio(
            shot_audio_ref,
            curr_idx=curr_idx,
            sched=sched,
            fps=fps,
            av_sync_lead_sec=float(av_sync_lead_sec),
            parent_ctx=parent_ctx,
            per_shot_audio_map=per_shot_audio_map,
            seed_ctx=seed_ctx,
        )

        proj = store.load_project()
        cur_rec = (proj.get("shots") or {}).get(str(curr_idx)) or {}
        old_active_tid = str(cur_rec.get("active_take_id") or "")
        active_meta = (cur_rec.get("takes") or {}).get(old_active_tid)
        if use_retry:
            rm = str(retry_lock.get("seed_mode") or "randomize").lower()
            rtxt = str(retry_lock.get("seed_text") or "").strip()
            if rm == "delta_+1":
                prev_s = str((active_meta or {}).get("seed_text") or "").strip()
                base_val = int(prev_s) if prev_s.isdigit() else int(shot_seed)
                resolved_seed = (base_val + 1) & 0xFFFFFFFFFFFFFFFF
                resolved_seed_text = str(resolved_seed)
            elif rm in ("fixed", "custom", "custom_string", "original"):
                resolved_seed, resolved_seed_text = _resolve_seed(
                    "custom_string" if rtxt else "original_retry", shot_seed, rtxt, active_meta
                )
            else:
                resolved_seed, resolved_seed_text = _resolve_seed("randomize", shot_seed, "", active_meta)
        else:
            resolved_seed, resolved_seed_text = _resolve_seed(seed_mode, shot_seed, seed_text, active_meta)

        trim_head = int(sched["trim_head_frames"])
        shifted_speech = [
            (int(s) + trim_head, int(e) + trim_head)
            for s, e in (shot.get("speech_spans") or ())
            if int(e) > int(s)
        ]

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
        loaded_imgs = mm_refs["images"]
        loaded_videos = mm_refs["videos"]
        loaded_voices = mm_refs["audios"]
        asset_ref_1 = loaded_imgs[0] if len(loaded_imgs) >= 1 else None
        asset_ref_2 = loaded_imgs[1] if len(loaded_imgs) >= 2 else None

        active_pipe = dict(shot_pipe)
        active_pipe.update({
            "current_shot_index": curr_idx,
            "current_shot": shot,
            "parent_take_id": parent_tid,
            "previous_active_take_id": old_active_tid,
            "keep_downstream_on_reroll": bool(keep_downstream_on_reroll),
            "is_continuation": is_cont,
            "continuation_depth": compute_continuation_depth(
                shots, curr_idx, is_cont, has_seed=bool(store.get_active_take_id(0))
            ),
            "target_frames": int(sched["target_frames"]),
            "trim_head_frames": trim_head,
            "keep_frames": int(sched["keep_frames"]),
            "overlap_frames": int(sched["overlap_frames"]),
            "overlap_gear": int(sched.get("overlap_gear") or shot_pipe.get("context_frames") or 22),
            "taper_tail_frames": int(sched.get("taper_tail_frames") or 22),
            "taper_ramp_frames": int(sched.get("taper_ramp_frames") or 4),
            "audio_guide_frames": int(sched.get("audio_guide_frames") or 0),
            "gear_feather_ticks": int(sched.get("audio_feather_ticks") or 7),
            "auto_scale_feather": True,
            "raw_speech_spans": list(shot.get("speech_spans") or ()),
            "speech_spans": shifted_speech,
            "seed_text": resolved_seed_text,
            "prompt": shot["prompt"],
            "prompt_sha256": shot["prompt_sha256"],
            "shot_type": int(shot["shot_type"]),
            "queue_remaining": remaining,
            "start_mode": start_mode,
            "delivery_ref_audio": delivery_ref_audio,
            "prev_audio": prev_audio if is_cont else _dummy_audio(),
            "prev_tail_latent": prev_tail_latent,
            "shot_audio_guide": shot_audio_guide,
            "parent_clean_tail_ref": prev_clean_images if is_cont else None,
            "scene_anchor_clean_ref": load_scene_anchor_clean_ref(store, shots, curr_idx, is_cont, parent_tid, prev_clean_images) if is_cont else None,
            "parent_audio_32k_ref": prev_audio if is_cont else None,
            "loaded_ref_images": loaded_imgs,
            "slot_to_ordinal": mm_refs.get("slot_to_ordinal") or {},
            "loaded_ref_videos": loaded_videos,
            "loaded_ref_audios": loaded_voices,
            "active_ref_summary": mm_refs["summary"],
            "loaded_asset_images": loaded_imgs,
            "loaded_asset_voices": loaded_voices,
            "asset_ref_count": len(loaded_imgs),
        })

        return (
            active_pipe,
            prev_clean_images,
            bool(shot.get("is_drama_mode", False)),
            shot["prompt"],
            width,
            height,
            int(sched["target_frames"]),
            resolved_seed,
            asset_ref_1,
            asset_ref_2,
        )
