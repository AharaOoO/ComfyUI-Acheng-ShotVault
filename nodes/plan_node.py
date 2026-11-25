"""[N1] ShotVaultPlan: Multi-format storyboard, BFS per-shot / MV audio slice, Shot 0 & asset turnaround planner."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch

from ..compat import resample_audio, resolve_output_root, sha256_text
from ..core.asset_and_turnaround import resolve_media_file, resolve_shot_assets, scan_assets_pool
from ..core.grid_math import ASPECT_PRESETS, AUDIO_VAE_SR, DELIVERY_AUDIO_SR, adapt_canvas_from_ratio, compute_cumulative_schedule, compute_tail_bridge_plan, resolve_duration_mode
from ..core.script_parser import parse_director_plan_json, parse_script
from ..core.seed_video_ingest import ingest_seed_video
from ..core.vault_store import VaultStore
from ..server_routes import _get_input_root, register_current_running_prompt

_AUD_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def _load_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """Decode audio file to `[2, samples]` float32 CPU tensor using PyAV."""
    import av  # type: ignore
    import numpy as np

    chunks: list[torch.Tensor] = []
    sr = DELIVERY_AUDIO_SR
    with av.open(path) as container:
        a_stream = next((s for s in container.streams if s.type == "audio"), None)
        if a_stream is None:
            raise ValueError(f"ShotVaultPlan: no audio stream in {path!r}")
        sr = int(a_stream.sample_rate or DELIVERY_AUDIO_SR)
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            nb_ch = int(getattr(frame.layout, "nb_channels", arr.shape[0] if arr.ndim > 1 else 1) or 1)
            if not frame.format.is_planar and nb_ch > 1:
                arr = arr.reshape(-1, nb_ch).T
            elif arr.ndim == 1:
                arr = arr[None, :]
            t = torch.from_numpy(np.ascontiguousarray(arr)).float()
            if t.abs().max() > 1.5:
                t = t / 32768.0
            if t.shape[0] == 1:
                t = t.repeat(2, 1)
            elif t.shape[0] > 2:
                t = t[:2]
            chunks.append(t)
    if not chunks:
        return torch.zeros((2, sr), dtype=torch.float32), sr
    return torch.cat(chunks, dim=1), sr


def _snap_boundaries_to_beats(waveform: torch.Tensor, sr: int, boundary_times: list[float], max_shift_sec: float = 0.25) -> list[float]:
    """Adjust internal cut timestamps to nearest RMS energy onset transient within `+/- max_shift_sec`."""
    if len(boundary_times) < 2 or waveform.numel() == 0:
        return boundary_times
    mono = waveform.mean(dim=0).float()
    hop = max(128, round(sr * 0.02))  # 20ms resolution
    n_frames = mono.numel() // hop
    if n_frames < 4:
        return boundary_times
    frames = mono[: n_frames * hop].view(n_frames, hop)
    rms = torch.sqrt((frames ** 2).mean(dim=1) + 1e-8)
    onset = torch.relu(rms[1:] - rms[:-1])

    snapped = [0.0]
    win = max(1, round(max_shift_sec / (hop / float(sr))))
    total_dur = mono.numel() / float(sr)
    for t in boundary_times[:-1]:
        center = max(1, min(onset.numel() - 2, round(t * sr / hop)))
        lo = max(1, center - win)
        hi = min(onset.numel() - 1, center + win + 1)
        if hi > lo:
            best_idx = lo + int(torch.argmax(onset[lo:hi]).item())
            best_t = max(snapped[-1] + 0.3, min(total_dur - 0.2, (best_idx + 1) * hop / float(sr)))
            snapped.append(best_t)
        else:
            snapped.append(t)
    snapped.append(boundary_times[-1])
    # Preserve order strictly: replace middle boundary times while keeping monotonic durations
    out = []
    prev = 0.0
    for i, t in enumerate(boundary_times[:-1]):
        cand = snapped[i + 1] if i + 1 < len(snapped) else t
        val = max(prev + 0.3, cand)
        out.append(val)
        prev = val
    out.append(max(prev + 0.3, boundary_times[-1]))
    return out


class ShotVaultPlan:
    """[N1] Parse storyboards, bind BFS per-shot or master MV audio, ingest Shot 0, and index assets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "MyH3Film"}),
                "script_format": (["auto", "1114_markdown", "kkdemabu_dual_jsonc", "cinema_pack_json"], {"default": "auto"}),
                "common_prompt": ("STRING", {"multiline": True, "default": ""}),
                "script_text": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (list(ASPECT_PRESETS.keys()), {"default": "16:9 (1344x768)"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.50, "step": 0.05}),
                "fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                "duration_mode": (["zero_tail_trim", "cumulative_cursor", "net_preserve"], {"default": "zero_tail_trim"}),
                "context_frames": (["22", "39", "56"], {"default": "22"}),
            },
            "optional": {
                "stage3_json_text": ("STRING", {"multiline": True, "default": ""}),
                "audio_drive_mode": (["none", "per_shot_audio_dir", "master_song_slice"], {"default": "none"}),
                "master_audio_in": ("AUDIO",),
                "audio_per_shot_dir": ("STRING", {"default": ""}),
                "snap_cuts_to_beat": ("BOOLEAN", {"default": False}),
                "seed_video_path": ("STRING", {"default": ""}),
                "seed_trim_end_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.04}),
                "assets_root_dir": ("STRING", {"default": ""}),
                "auto_split_turnaround": ("BOOLEAN", {"default": False}),
                "director_plan_json": ("STRING", {"default": ""}),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("SHOT_PIPE", "INT", "STRING")
    RETURN_NAMES = ("shot_pipe", "total_shots", "plan_summary_json")
    FUNCTION = "plan"
    CATEGORY = "Acheng-ShotVault/Core"

    def plan(
        self,
        project_name: str,
        script_format: str,
        common_prompt: str,
        script_text: str,
        aspect_ratio: str,
        megapixels: float,
        fps: int,
        duration_mode: str,
        context_frames: str,
        stage3_json_text: str = "",
        audio_drive_mode: str = "none",
        master_audio_in: Optional[dict] = None,
        audio_per_shot_dir: str = "",
        snap_cuts_to_beat: bool = False,
        seed_video_path: str = "",
        seed_trim_end_sec: float = 0.0,
        assets_root_dir: str = "",
        auto_split_turnaround: bool = False,
        director_plan_json: str = "",
        vae=None,
        audio_vae=None,
    ):
        project_root = resolve_output_root(project_name)
        store = VaultStore(project_root)
        register_current_running_prompt()
        width, height = adapt_canvas_from_ratio(aspect_ratio, megapixels)
        input_root = _get_input_root()

        global_refs_cfg, director_shots = parse_director_plan_json(
            director_plan_json=director_plan_json,
            common_prompt=common_prompt,
            fps=int(fps),
            has_seed_video=bool((seed_video_path or "").strip()),
        )

        raw_seed_path = (seed_video_path or "").strip() or str(global_refs_cfg.get("seed_video_path") or "").strip()
        effective_seed_trim = float(seed_trim_end_sec) if float(seed_trim_end_sec) > 0.0 else float(global_refs_cfg.get("seed_trim_end_sec") or 0.0)
        seed_path = resolve_media_file(raw_seed_path, input_root, project_root) if raw_seed_path else ""
        has_seed = bool(seed_path and os.path.isfile(seed_path))
        if has_seed:
            proj = store.load_project()
            cur_seed = proj.get("seed_shot") or {}
            seed_ctx = store.load_take_context("shot_0000_seed")
            same_seed = (
                cur_seed.get("source_path") == seed_path
                and abs(float(cur_seed.get("trim_end_sec", -1.0)) - effective_seed_trim) < 1e-4
                and seed_ctx is not None
                and int(seed_ctx["meta"].get("width", 0)) == width
                and int(seed_ctx["meta"].get("height", 0)) == height
                and (vae is None or bool(seed_ctx["meta"].get("has_tail_latent", False)))
            )
            if not same_seed:
                ingest_seed_video(
                    store,
                    seed_video_path=seed_path,
                    seed_trim_end_sec=effective_seed_trim,
                    width=width,
                    height=height,
                    fps=int(fps),
                    vae=vae,
                    audio_vae=audio_vae,
                )
                proj = store.load_project()
                if proj.get("seed_shot"):
                    proj["seed_shot"]["trim_end_sec"] = effective_seed_trim
                    store.save_project(proj)

        default_dur_cfg = float(global_refs_cfg.get("default_duration_sec") or 5.0)
        has_ui_audio_key = isinstance(global_refs_cfg, dict) and ("audio_drive_mode" in global_refs_cfg)
        ui_audio_mode = str(global_refs_cfg.get("audio_drive_mode") or "").strip()
        if has_ui_audio_key and ui_audio_mode in ("none", "per_shot_audio_dir", "master_song_slice"):
            effective_audio_mode = ui_audio_mode
        else:
            effective_audio_mode = audio_drive_mode if audio_drive_mode in ("per_shot_audio_dir", "master_song_slice") else "none"
        explicit_regular_mode = (effective_audio_mode == "none" and has_ui_audio_key and ui_audio_mode == "none")

        if director_shots:
            shots = director_shots
            if has_seed and not shots[0].get("explicit_hard_cut"):
                shots[0]["is_continuation"] = True
            elif not has_seed:
                shots[0]["is_continuation"] = False
        else:
            shots = parse_script(
                script_text=script_text,
                common_prompt=common_prompt,
                stage3_json_text=stage3_json_text,
                script_format=script_format,
                fps=int(fps),
                has_seed_video=has_seed,
                default_duration_sec=default_dur_cfg,
            )

        # Backup script & structured director plan into project script/ folder
        with open(os.path.join(store.script_dir, "storyboard_raw.txt"), "w", encoding="utf-8") as f:
            f.write(script_text or "")
        if (director_plan_json or "").strip():
            with open(os.path.join(store.script_dir, "director_plan.json"), "w", encoding="utf-8") as f:
                f.write(director_plan_json)

        # Audio binding: explicit per-shot `audio_file` (from Director UI or @audio_file), per_shot_audio_dir, or master_song_slice
        dir_audio_files: list[str] = []
        if audio_per_shot_dir and os.path.isdir(audio_per_shot_dir):
            dir_audio_files = [
                os.path.join(audio_per_shot_dir, fn)
                for fn in sorted(os.listdir(audio_per_shot_dir))
                if fn.lower().endswith(_AUD_EXTS)
            ]

        has_master_audio = (
            isinstance(master_audio_in, dict)
            and isinstance(master_audio_in.get("waveform"), torch.Tensor)
            and master_audio_in["waveform"].numel() > 0
        )
        per_shot_audio_tensors: dict[int, dict[str, Any]] = {}
        has_explicit_shot_audio = (not explicit_regular_mode) and any(bool(sh.get("audio_file")) for sh in shots)
        use_per_shot_fallback = False
        if effective_audio_mode == "master_song_slice" and not has_master_audio:
            if any(bool(sh.get("audio_file")) for sh in shots) or bool(dir_audio_files):
                use_per_shot_fallback = True
                effective_audio_mode = "per_shot_audio_dir"
            else:
                raise ValueError(
                    "ShotVault: 当前开启了『🎵 音频驱动 · 整曲母带切片』，但左下角 Group 4『开关 · 整曲母带音频』"
                    "仍处于旁路状态（未接入母带音频）！请点亮该开关，或在分镜规划顶栏切回『🎬 常规模式』。\n"
                    "(EN) 'Audio-Driven · Master Track Slice' is enabled, but master_audio_in is bypassed/unconnected. "
                    "Please un-bypass the master audio node or switch back to 'Regular' mode."
                )
        audio_drive_active = (effective_audio_mode in ("per_shot_audio_dir", "master_song_slice")) or has_explicit_shot_audio
        if use_per_shot_fallback or effective_audio_mode == "per_shot_audio_dir" or (effective_audio_mode == "none" and has_explicit_shot_audio):
            for i, sh in enumerate(shots):
                raw_ap = sh.get("audio_file") or ""
                apath = resolve_media_file(raw_ap, input_root, project_root) if raw_ap else ""
                if (not apath or not os.path.isfile(apath)) and raw_ap and not os.path.isabs(raw_ap) and audio_per_shot_dir:
                    cand = os.path.join(audio_per_shot_dir, raw_ap)
                    if os.path.isfile(cand):
                        apath = cand
                if (not apath or not os.path.isfile(apath)) and effective_audio_mode == "per_shot_audio_dir" and i < len(dir_audio_files):
                    apath = dir_audio_files[i]
                if apath and os.path.isfile(apath):
                    wav, sr = _load_audio_file(apath)
                    dur = wav.shape[1] / float(sr)
                    sh["duration_sec"] = max(0.25, dur)
                    per_shot_audio_tensors[sh["shot_index"]] = {
                        "waveform": wav.unsqueeze(0),
                        "sample_rate": sr,
                        "sha256": sha256_text(f"{apath}:{wav.shape[1]}"),
                    }
        elif effective_audio_mode == "master_song_slice" and has_master_audio:
            mw = master_audio_in["waveform"][0].detach().cpu().float()
            msr = int(master_audio_in.get("sample_rate", 48000))
            if mw.shape[0] == 1:
                mw = mw.repeat(2, 1)
            elif mw.shape[0] > 2:
                mw = mw[:2]
            boundaries = []
            acc = 0.0
            for sh in shots:
                t1 = float(sh["t_end_sec"]) if sh.get("t_end_sec") is not None else (acc + float(sh["duration_sec"]))
                boundaries.append(t1)
                acc = t1
            if snap_cuts_to_beat:
                boundaries = _snap_boundaries_to_beats(mw, msr, boundaries)
            span_times: list[tuple[float, float]] = []
            t0 = 0.0
            for sh, t1 in zip(shots, boundaries):
                sh["duration_sec"] = max(0.25, t1 - t0)
                span_times.append((t0, t1))
                t0 = t1
            sched_prev = compute_cumulative_schedule(
                [s["duration_sec"] for s in shots],
                fps=int(fps),
                is_continuations=[s["is_continuation"] for s in shots],
                context_frames=39,
                duration_mode=resolve_duration_mode(duration_mode, True),
            )
            seed_tail_aud: Optional[torch.Tensor] = None
            if has_seed:
                s_ctx = store.load_take_context("shot_0000_seed")
                if isinstance(s_ctx, dict):
                    raw_sd = (s_ctx.get("tensors") or {}).get("delivery_audio_48k")
                    if isinstance(raw_sd, torch.Tensor) and raw_sd.numel() > 0:
                        seed_tail_aud = resample_audio(raw_sd.detach().cpu().float(), DELIVERY_AUDIO_SR, msr)
            fps_f = float(max(1, int(fps)))
            for sh, sc in zip(shots, sched_prev):
                f_start = int(sc["cum_frame_start"])
                f_end = int(sc["cum_frame_end"])
                s0 = max(0, round((f_start / fps_f) * msr))
                s1 = min(mw.shape[1], round((f_end / fps_f) * msr))
                sl = mw[:, s0:s1]
                if sl.shape[1] < 64:
                    sl = torch.zeros((2, max(64, round(int(sc["keep_frames"]) / fps_f * msr))), dtype=torch.float32)
                g_start_frame = f_start - int(sc["trim_head_frames"])
                need_g = max(1, round((int(sc["target_frames"]) / fps_f) * msr))
                if g_start_frame < 0:
                    pre_need = round(((-g_start_frame) / fps_f) * msr)
                    if seed_tail_aud is not None and seed_tail_aud.shape[1] > 0:
                        pre_sl = seed_tail_aud[:, -pre_need:] if seed_tail_aud.shape[1] >= pre_need else torch.nn.functional.pad(seed_tail_aud, (pre_need - seed_tail_aud.shape[1], 0))
                    else:
                        pre_sl = torch.zeros((2, pre_need), dtype=torch.float32)
                    fwd_sl = mw[:, 0:min(mw.shape[1], need_g - pre_need)]
                    g_sl = torch.cat([pre_sl, fwd_sl], dim=1)
                else:
                    gs0 = max(0, round((g_start_frame / fps_f) * msr))
                    g_sl = mw[:, gs0:min(mw.shape[1], gs0 + need_g)]
                if g_sl.shape[1] < need_g:
                    g_sl = torch.nn.functional.pad(g_sl, (0, need_g - g_sl.shape[1]))
                per_shot_audio_tensors[sh["shot_index"]] = {
                    "waveform": sl.unsqueeze(0).contiguous(),
                    "sample_rate": msr,
                    "guide_waveform": g_sl.unsqueeze(0).contiguous(),
                    "guide_sample_rate": msr,
                    "guide_prealigned": True,
                    "sha256": sha256_text(f"master:{s0}:{s1}:{need_g}"),
                }

        # Assets indexing & turnaround prompt hint injection
        pool = scan_assets_pool(assets_root_dir)
        for sh in shots:
            bound = resolve_shot_assets(pool, sh["prompt"], auto_split_turnaround=bool(auto_split_turnaround))
            sh["assets"] = bound
            if bound["turnaround_hint"]:
                sh["prompt"] = f"{sh['prompt']}\n{bound['turnaround_hint']}"
                sh["prompt_sha256"] = sha256_text(sh["prompt"])

        proj = store.load_project()
        proj["last_plan"] = [
            {
                "shot_index": s["shot_index"],
                "shot_id": s["shot_id"],
                "duration_sec": s["duration_sec"],
                "shot_type": s["shot_type"],
                "is_continuation": s["is_continuation"],
                "prompt_sha256": s["prompt_sha256"],
                "local_prompt": s["local_prompt"],
            }
            for s in shots
        ]
        proj["planned_shots"] = proj["last_plan"]
        proj["fps"] = int(fps)
        proj["width"] = int(width)
        proj["height"] = int(height)
        store.save_project(proj)

        if audio_drive_active:
            effective_context_frames = 39
            for sh in shots:
                sh["overlap_override"] = 39 if sh["is_continuation"] else 0
            overlap_overrides = [39 if s["is_continuation"] else 0 for s in shots]
        else:
            effective_context_frames = int(global_refs_cfg.get("context_frames") or 0) or int(context_frames)
            if effective_context_frames not in (22, 39, 56):
                effective_context_frames = 22
            overlap_overrides = [int(s.get("overlap_override") or 0) for s in shots]
        retry_lock = dict(global_refs_cfg.get("retry_lock") or {})
        effective_duration_mode = resolve_duration_mode(duration_mode, audio_drive_active)
        sched_summary = compute_cumulative_schedule(
            [s["duration_sec"] for s in shots] or [5.0],
            fps=int(fps),
            is_continuations=[s["is_continuation"] for s in shots] or [False],
            context_frames=effective_context_frames,
            duration_mode=effective_duration_mode,
            overlap_overrides=overlap_overrides,
        )
        per_shot_tf = [int(sc["target_frames"]) for sc in sched_summary] or [124]
        if bool(retry_lock.get("enabled")):
            r_idx = max(1, min(len(per_shot_tf), int(retry_lock.get("shot_index") or 1)))
            tb_req = int(retry_lock.get("tail_bridge_frames") or 0)
            if tb_req in (1, 5, 22) and r_idx < len(shots) and store.get_active_take_id(r_idx + 1):
                r_sc = sched_summary[r_idx - 1]
                tb_plan = compute_tail_bridge_plan(
                    r_sc["trim_head_frames"], r_sc["keep_frames"], tb_req, audio_drive_active=audio_drive_active
                )
                per_shot_tf[r_idx - 1] = int(tb_plan["target_frames"])
            preview_target_frames = per_shot_tf[r_idx - 1]
        else:
            preview_target_frames = max(per_shot_tf)
        out_audio_mode = (
            effective_audio_mode if effective_audio_mode != "none"
            else ("per_shot_audio_dir" if has_explicit_shot_audio else "none")
        )

        anti_degradation = dict(
            global_refs_cfg.get("anti_degradation")
            or {"mode": "standard", "noise_keep_ratio": 0.40, "static_bg_pull": 0.20}
        )
        shot_pipe = {
            "project_name": proj["project_name"],
            "project_root": project_root,
            "common_prompt": common_prompt or "",
            "fps": int(fps),
            "width": int(width),
            "height": int(height),
            "duration_mode": effective_duration_mode,
            "context_frames": effective_context_frames,
            "overlap_overrides": overlap_overrides,
            "retry_lock": retry_lock,
            "anti_degradation": anti_degradation,
            "audio_drive_mode": out_audio_mode,
            "audio_drive_active": audio_drive_active,
            "auto_split_turnaround": bool(auto_split_turnaround),
            "global_refs_cfg": global_refs_cfg,
            "shots": shots,
            "per_shot_audio": per_shot_audio_tensors,
        }
        loop_total = 1 if bool(retry_lock.get("enabled")) else len(shots)
        summary = json.dumps(
            {
                "project_root": project_root,
                "width": width,
                "height": height,
                "fps": int(fps),
                "context_frames": effective_context_frames,
                "audio_drive_active": audio_drive_active,
                "audio_drive_mode": out_audio_mode,
                "loop_total": loop_total,
                "total_planned_shots": len(shots),
                "preview_target_frames": int(preview_target_frames),
                "target_frames_per_shot": per_shot_tf,
                "shots": proj["last_plan"],
            },
            ensure_ascii=False,
            indent=2,
        )
        return (shot_pipe, loop_total, summary)
