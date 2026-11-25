"""[N5] ShotVaultSave: Dual clean/final storage isolation, 12-step bare latent tail archival & safe queue-head advance."""

from __future__ import annotations

import gc
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F

from ..compat import resample_audio
from ..core.grid_math import DELIVERY_AUDIO_SR
from ..core.seam_qc import analyze_seam_diagnostics, build_seam_qc_clip, write_jpg_thumb, write_mp4_with_pcm
from ..core.vault_store import VaultStore
from ..server_routes import is_aborted, schedule_next_queue_head


def _wrap_video_file(video_abs: str):
    """Wrap an already-encoded MP4/MOV on disk as a ComfyUI `VIDEO` (`VideoFromFile`) with zero re-encoding."""
    if not video_abs or not os.path.isfile(video_abs):
        return None
    try:
        from comfy_api.latest import InputImpl  # type: ignore

        return InputImpl.VideoFromFile(video_abs)
    except Exception:
        return None


def _extract_stereo_48k(audio_dict: Optional[dict], target_frames: int, fps: int) -> Optional[torch.Tensor]:
    if not isinstance(audio_dict, dict) or not isinstance(audio_dict.get("waveform"), torch.Tensor):
        return None
    w = audio_dict["waveform"].detach().cpu().float()
    if w.ndim == 3:
        w = w[0]
    elif w.ndim == 1:
        w = w.unsqueeze(0)
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    elif w.shape[0] > 2:
        w = w[:2]
    w48 = resample_audio(w, int(audio_dict.get("sample_rate", 48000)), DELIVERY_AUDIO_SR)
    need = max(1, round(int(target_frames) / float(fps) * DELIVERY_AUDIO_SR))
    if w48.shape[1] < need:
        w48 = F.pad(w48, (0, need - w48.shape[1]))
    else:
        w48 = w48[:, :need]
    return w48.contiguous()


class ShotVaultSave:
    """[N5] Archive upscaled MP4 & un-upscaled clean `.h3ctx` + tail latent separately; queue next shot at queue head."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_images": ("IMAGE",),
                "shot_pipe": ("SHOT_PIPE",),
                "auto_queue_next": ("BOOLEAN", {"default": True}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "codec": (["h264", "prores_ks"], {"default": "h264"}),
            },
            "optional": {
                "final_audio": ("AUDIO",),
                "clean_images": ("IMAGE",),
                "sampled_latent": ("LATENT",),
                "delivery_audio_override": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "SHOT_PIPE", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("final_images", "audio", "updated_pipe", "take_id", "video_path", "video")
    FUNCTION = "save_take"
    CATEGORY = "Acheng-ShotVault/Core"
    OUTPUT_NODE = True

    def save_take(
        self,
        final_images: torch.Tensor,
        shot_pipe: dict,
        auto_queue_next: bool,
        crf: int,
        codec: str,
        final_audio: Optional[dict] = None,
        clean_images: Optional[torch.Tensor] = None,
        sampled_latent: Optional[dict] = None,
        delivery_audio_override: Optional[dict] = None,
    ):
        project_root = str(shot_pipe["project_root"])
        store = VaultStore(project_root)
        fps = int(shot_pipe.get("fps", 24))
        shot_idx = int(shot_pipe.get("current_shot_index", 1))
        n_frames = int(final_images.shape[0])

        ctx_images = clean_images if (isinstance(clean_images, torch.Tensor) and clean_images.shape[0] > 0) else final_images
        wav_48k = _extract_stereo_48k(delivery_audio_override, n_frames, fps)
        if wav_48k is None:
            wav_48k = _extract_stereo_48k(shot_pipe.get("delivery_ref_audio"), n_frames, fps)
        if wav_48k is None:
            wav_48k = _extract_stereo_48k(final_audio, n_frames, fps)
        if wav_48k is None:
            wav_48k = torch.zeros((2, max(1, round(n_frames / float(fps) * DELIVERY_AUDIO_SR))), dtype=torch.float32)

        out_audio = {"waveform": wav_48k.unsqueeze(0), "sample_rate": DELIVERY_AUDIO_SR}

        take_id, lock_path = store.reserve_next_take(shot_idx)
        try:
            ext = "mov" if codec == "prores_ks" else "mp4"
            video_rel = f"shots/{take_id}.{ext}"
            video_abs = os.path.join(project_root, video_rel)
            thumb_rel = f"_vault/thumb_{take_id}.jpg"
            thumb_abs = os.path.join(project_root, thumb_rel)

            write_mp4_with_pcm(final_images, wav_48k, DELIVERY_AUDIO_SR, fps, video_abs, crf=int(crf), codec=codec)
            write_jpg_thumb(final_images[0], thumb_abs)

            parent_tid = str(shot_pipe.get("parent_take_id") or "")
            seam_rel = ""
            seam_diag: Optional[dict] = None
            if parent_tid:
                p_ctx = store.load_take_context(parent_tid)
                if p_ctx is not None and "clean_tail" in p_ctx["tensors"]:
                    p_frames = p_ctx["tensors"]["clean_tail"].float()
                    p_aud = p_ctx["tensors"].get("delivery_audio_48k")
                    if p_aud is None:
                        p_aud = p_ctx["tensors"].get("audio_32k_tail")
                        if p_aud is not None:
                            p_aud = resample_audio(p_aud.float(), 32000, DELIVERY_AUDIO_SR)
                    seam_rel = f"_vault/seam_{take_id}.mp4"
                    build_seam_qc_clip(
                        p_frames,
                        p_aud,
                        ctx_images,
                        wav_48k,
                        fps,
                        os.path.join(project_root, seam_rel),
                    )
                    seam_diag = analyze_seam_diagnostics(p_frames, ctx_images, fps=fps)
            st_rep = shot_pipe.get("last_seam_trim_report")
            if isinstance(st_rep, dict):
                seam_diag = dict(seam_diag or {})
                seam_diag["seam_trim_v53"] = st_rep

            txt_abs = os.path.join(project_root, f"shots/{take_id}.txt")
            os.makedirs(os.path.dirname(txt_abs), exist_ok=True)
            mode_label = "1116文戏" if int(shot_pipe.get("shot_type", 1)) == 0 else "1114武戏"
            sidecar_lines = [
                f"project={shot_pipe.get('project_name', '')}",
                f"shot_index={shot_idx} | take_id={take_id} | mode={mode_label}",
                f"seed={shot_pipe.get('seed_text', '0')} | canvas={int(final_images.shape[2])}x{int(final_images.shape[1])}@{fps}fps",
                f"gear_overlap={int(shot_pipe.get('overlap_frames') or 0)}f | trim_head={int(shot_pipe.get('trim_head_frames') or 0)}f | keep_frames={int(shot_pipe.get('keep_frames') or n_frames)}f",
                f"video_file={video_rel}",
                f"refs={shot_pipe.get('active_ref_summary', '')}",
                f"seam_trim_v53={json.dumps(st_rep, ensure_ascii=False) if isinstance(st_rep, dict) else '{}'}",
                "",
                "[PROMPT]",
                str(shot_pipe.get("prompt") or ""),
            ]
            with open(txt_abs, "w", encoding="utf-8") as sf:
                sf.write("\n".join(sidecar_lines) + "\n")

            parent_clean_ref = shot_pipe.get("parent_clean_tail_ref")
            tail_lat_target = max(39, int(shot_pipe.get("overlap_frames") or 39))
            store.save_take_context(
                take_id=take_id,
                shot_index=shot_idx,
                clean_images=ctx_images,
                audio_waveform_48k=wav_48k,
                audio_sr=DELIVERY_AUDIO_SR,
                fps=fps,
                prompt_sha256=str(shot_pipe.get("prompt_sha256") or ""),
                parent_take_id=parent_tid,
                seed_text=str(shot_pipe.get("seed_text") or "0"),
                keep_frames=int(shot_pipe.get("keep_frames") or n_frames),
                trim_head_frames=int(shot_pipe.get("trim_head_frames") or 0),
                shot_type=int(shot_pipe.get("shot_type", 1)),
                parent_clean_tail=parent_clean_ref if isinstance(parent_clean_ref, torch.Tensor) else None,
                sampled_latent=sampled_latent,
                tail_latent_frames=min(56, tail_lat_target),
                video_rel_path=video_rel,
                thumb_rel_path=thumb_rel,
                seam_rel_path=seam_rel,
                seam_diagnostics=seam_diag,
                set_active=True,
                raw_trimmed_frames=int(ctx_images.shape[0]),
            )
            # Scoped reroll downstream protection: clear this shot's own accepted badge and scope shot_idx+1 only.
            proj = store.load_project()
            cur_rec = (proj.get("shots") or {}).get(str(shot_idx))
            if isinstance(cur_rec, dict):
                cur_rec.pop("accepted_parent_take_id", None)
            old_active = str(shot_pipe.get("previous_active_take_id") or "")
            keep_ds = bool(shot_pipe.get("keep_downstream_on_reroll", False))
            bridged_next_tid = str(shot_pipe.get("bridged_next_take_id") or "")
            if bridged_next_tid and isinstance(cur_rec, dict):
                t_rec = (cur_rec.get("takes") or {}).get(take_id)
                if isinstance(t_rec, dict):
                    t_rec["bridged_next_take_id"] = bridged_next_tid
                    t_rec["tail_bridge_frames"] = int(shot_pipe.get("tail_bridge_frames") or 0)
            next_rec = (proj.get("shots") or {}).get(str(shot_idx + 1))
            if isinstance(next_rec, dict):
                next_active_id = str(next_rec.get("active_take_id") or "")
                next_meta = (next_rec.get("takes") or {}).get(next_active_id) or {}
                next_parent = str(next_meta.get("parent_take_id") or "")
                prev_accepted = str(next_rec.get("accepted_parent_take_id") or "")
                if bridged_next_tid and next_active_id == bridged_next_tid:
                    # Bidirectional bridge explicitly anchored to next_active_id -> keep downstream green VALID!
                    next_rec["accepted_parent_take_id"] = take_id
                elif keep_ds and old_active and old_active != take_id and (next_parent == old_active or prev_accepted == old_active):
                    next_rec["accepted_parent_take_id"] = take_id
                else:
                    next_rec.pop("accepted_parent_take_id", None)
            store.save_project(proj)
        finally:
            store.release_reserve(lock_path)

        # Optional: auto-refresh master.mp4 immediately after single-shot reroll (allow_with_warning)
        retry_cfg = shot_pipe.get("retry_lock") if isinstance(shot_pipe.get("retry_lock"), dict) else {}
        auto_assembled_master = ""
        if bool(retry_cfg.get("enabled")) and bool(retry_cfg.get("auto_assemble_after_reroll", False)):
            try:
                from ..core.assembler import assemble_master

                shots_plan = list(shot_pipe.get("shots") or ())
                if not shots_plan:
                    pj_now = store.load_project()
                    shots_plan = list(pj_now.get("last_plan") or pj_now.get("planned_shots") or ())
                if shots_plan:
                    auto_assembled_master, _, _ = assemble_master(
                        store=store,
                        shots_plan=shots_plan,
                        fps=fps,
                        stale_policy="allow_with_warning",
                        export_fcpxml=True,
                    )
            except Exception:
                auto_assembled_master = ""

        remaining = list(shot_pipe.get("queue_remaining") or ())
        queued_next = False
        if auto_queue_next and remaining and not is_aborted():
            queued_next = schedule_next_queue_head(remaining)

        updated_pipe = dict(shot_pipe)
        for _tkey in (
            "loaded_ref_images",
            "loaded_ref_videos",
            "loaded_ref_audios",
            "loaded_asset_images",
            "loaded_asset_voices",
            "prev_tail_latent",
            "parent_clean_tail_ref",
            "scene_anchor_clean_ref",
            "parent_audio_32k_ref",
            "prev_audio",
            "shot_audio_guide",
            "delivery_ref_audio",
            "next_head_clean_ref",
            "next_head_audio_32k",
            "_cached_head_video_latent",
            "_cached_head_audio_kfs",
            "_cached_tail_video_latent",
            "_cached_tail_audio_latent",
        ):
            updated_pipe.pop(_tkey, None)
        updated_pipe["last_saved_take_id"] = take_id
        updated_pipe["last_saved_video"] = video_abs
        updated_pipe["queued_next"] = queued_next
        if auto_assembled_master:
            updated_pipe["auto_assembled_master"] = auto_assembled_master
        gc.collect()

        video_handle = _wrap_video_file(video_abs)
        return {
            "ui": {
                "shotvault_saved": [{
                    "project": shot_pipe.get("project_name", ""),
                    "shot_index": shot_idx,
                    "take_id": take_id,
                    "queued_next": queued_next,
                    "remaining": remaining,
                }]
            },
            "result": (final_images, out_audio, updated_pipe, take_id, video_abs, video_handle),
        }
