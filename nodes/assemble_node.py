"""[N6] ShotVaultAssemble: Lineage DAG check, lossless concat, single-pass f32le PCM mux, and DaVinci FCPXML export."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from ..compat import resolve_output_root
from ..core.assembler import assemble_master
from ..core.vault_store import VaultStore


def _wrap_master_video(video_abs: str):
    """Wrap assembled master MP4 as ComfyUI `VIDEO` (`VideoFromFile`) with zero re-encoding."""
    if not video_abs or not os.path.isfile(video_abs):
        return None
    try:
        from comfy_api.latest import InputImpl  # type: ignore

        return InputImpl.VideoFromFile(video_abs)
    except Exception:
        return None


class ShotVaultAssemble:
    """[N6] Enforce STALE lineage gate (or auto-defer mid-run) and export single-pass PCM master + `.fcpxml`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "MyH3Film"}),
                "stale_policy": (["defer_until_complete", "block_and_report", "allow_with_warning"], {"default": "defer_until_complete"}),
                "audio_source_policy": (["auto_prefer_reference", "generated_only", "master_override"], {"default": "auto_prefer_reference"}),
                "audio_crossfade_ms": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "post_mux_offset_ms": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0, "step": 5.0}),
                "prepend_seed_video": ("BOOLEAN", {"default": True}),
                "export_fcpxml": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "shot_pipe": ("SHOT_PIPE",),
                "master_audio_override": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("master_path", "fcpxml_path", "report_json", "master_video")
    FUNCTION = "assemble"
    CATEGORY = "Acheng-ShotVault/Core"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def assemble(
        self,
        project_name: str,
        stale_policy: str,
        audio_source_policy: str,
        audio_crossfade_ms: float,
        post_mux_offset_ms: float,
        prepend_seed_video: bool,
        export_fcpxml: bool,
        shot_pipe: Optional[dict] = None,
        master_audio_override: Optional[dict] = None,
    ):
        if isinstance(shot_pipe, dict) and shot_pipe.get("project_root"):
            project_root = str(shot_pipe["project_root"])
            shots_plan = list(shot_pipe.get("shots") or ())
            fps = int(shot_pipe.get("fps", 24))
            store = VaultStore(project_root)
        else:
            project_root = resolve_output_root(project_name)
            store = VaultStore(project_root)
            proj = store.load_project()
            shots_plan = list(proj.get("last_plan") or ())
            if not shots_plan and proj.get("shots"):
                shots_plan = [
                    {
                        "shot_index": int(k),
                        "shot_id": f"shot_{int(k):04d}",
                        "is_continuation": int(k) > 1,
                        "prompt_sha256": (v.get("takes", {}).get(v.get("active_take_id", ""), {}) or {}).get("prompt_sha256", ""),
                    }
                    for k, v in sorted(proj["shots"].items(), key=lambda kv: int(kv[0]))
                ]
            fps = int(proj.get("fps", 24))

        has_pending_queue = bool(
            isinstance(shot_pipe, dict) and (shot_pipe.get("queued_next") or shot_pipe.get("queue_remaining"))
        )
        master_mp4, fcpxml_path, report_json = assemble_master(
            store=store,
            shots_plan=shots_plan,
            fps=fps,
            stale_policy=stale_policy,
            audio_source_policy=audio_source_policy,
            master_audio_override=master_audio_override,
            audio_crossfade_ms=float(audio_crossfade_ms),
            post_mux_offset_ms=float(post_mux_offset_ms),
            prepend_seed_video=bool(prepend_seed_video),
            export_fcpxml=bool(export_fcpxml),
            has_pending_queue=has_pending_queue,
        )
        ui_key = "shotvault_assembled" if master_mp4 else "shotvault_deferred"
        ui_payload = (
            [{"master": master_mp4, "fcpxml": fcpxml_path}]
            if master_mp4
            else [json.loads(report_json)]
        )
        return {
            "ui": {ui_key: ui_payload},
            "result": (master_mp4, fcpxml_path, report_json, _wrap_master_video(master_mp4)),
        }
