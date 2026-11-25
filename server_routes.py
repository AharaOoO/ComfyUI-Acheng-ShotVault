"""PromptServer HTTP routes, Director Sidebar API, and Queue-Head auto-advance scheduler."""

from __future__ import annotations

import asyncio
import copy
import datetime
import json
import logging
import os
import threading
import uuid
from collections import deque
from typing import Any, Optional
from urllib.parse import quote

from .compat import ensure_inside_root, resolve_output_root, sanitize_project_name
from .core.lineage_dag import LineageDAG
from .core.script_parser import parse_director_plan_json, parse_script
from .core.vault_store import VaultStore, compute_master_sync_status

LOGGER = logging.getLogger("ShotVault.Server")

_LOCK = threading.Lock()
_ASSEMBLE_LOCKS_GUARD = threading.Lock()
_ASSEMBLE_LOCKS: dict[str, threading.Lock] = {}
_ABORT_AUTO_QUEUE = False
_ROUTES_REGISTERED = False
_SHOTVAULT_PROMPT_IDS: deque[str] = deque(maxlen=256)


def _get_assemble_lock(project_root: str) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(project_root))
    with _ASSEMBLE_LOCKS_GUARD:
        lk = _ASSEMBLE_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _ASSEMBLE_LOCKS[key] = lk
        return lk


def _scan_master_exports(project_root: str) -> list[dict[str, Any]]:
    """List existing exported master `.mp4`/`.fcpxml` artifacts in `master/` newest-first."""
    mdir = os.path.join(project_root, "master")
    if not os.path.isdir(mdir):
        return []
    out: list[dict[str, Any]] = []
    for fn in os.listdir(mdir):
        low = fn.lower()
        if not low.endswith((".mp4", ".mov", ".fcpxml")) or ".tmp." in low:
            continue
        ap = os.path.join(mdir, fn)
        if not os.path.isfile(ap):
            continue
        try:
            st = os.stat(ap)
            out.append({
                "name": fn,
                "rel": f"master/{fn}",
                "ext": os.path.splitext(fn)[1].lower(),
                "bytes": int(st.st_size),
                "mtime": float(st.st_mtime),
            })
        except OSError:
            continue
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out[:20]


def handle_assemble_request(data: dict[str, Any]) -> dict[str, Any]:
    """Thread-safe full-film assembly endpoint worker (no GPU / no diffusion re-run)."""
    from .core.assembler import assemble_master

    project = str(data.get("project") or "MyH3Film")
    stale_policy = str(data.get("stale_policy") or "block_and_report")
    if stale_policy not in ("allow_with_warning", "block_and_report", "defer_until_complete"):
        stale_policy = "allow_with_warning"
    audio_source_policy = str(data.get("audio_source_policy") or "auto_prefer_reference")
    if audio_source_policy not in ("auto_prefer_reference", "generated_only", "master_override"):
        audio_source_policy = "auto_prefer_reference"
    try:
        audio_crossfade_ms = max(0.0, min(200.0, float(data.get("audio_crossfade_ms", 15.0))))
    except (TypeError, ValueError):
        audio_crossfade_ms = 15.0
    prepend_seed_video = bool(data.get("prepend_seed_video", True))
    export_fcpxml = bool(data.get("export_fcpxml", True))
    output_mode = str(data.get("output_mode") or "overwrite_master")
    custom_name = sanitize_project_name(str(data.get("master_filename") or "").strip()) if str(data.get("master_filename") or "").strip() else ""

    root = resolve_output_root(project)
    lock = _get_assemble_lock(root)
    with lock:
        store = VaultStore(root)
        proj = store.load_project()
        shots_plan = list(proj.get("last_plan") or proj.get("planned_shots") or ())
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
        fps = int(proj.get("fps") or 24)
        proj_name = sanitize_project_name(str(proj.get("project_name") or project))
        if custom_name:
            base_name = custom_name
        elif output_mode == "timestamped":
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{proj_name}_master_{ts}"
        else:
            base_name = f"{proj_name}_master"

        try:
            master_mp4, fcpxml_path, report_json = assemble_master(
                store=store,
                shots_plan=shots_plan,
                fps=fps,
                stale_policy=stale_policy,
                audio_source_policy=audio_source_policy,
                audio_crossfade_ms=audio_crossfade_ms,
                prepend_seed_video=prepend_seed_video,
                export_fcpxml=export_fcpxml,
                master_filename=base_name,
            )
            rep_obj = {}
            try:
                rep_obj = json.loads(report_json)
            except Exception:
                rep_obj = {}
            master_rel = f"master/{os.path.basename(master_mp4)}" if master_mp4 else ""
            fcpxml_rel = f"master/{os.path.basename(fcpxml_path)}" if fcpxml_path else ""
            file_bytes = int(os.path.getsize(master_mp4)) if (master_mp4 and os.path.isfile(master_mp4)) else 0
            return {
                "ok": bool(master_mp4),
                "master_mp4": master_mp4,
                "master_rel": master_rel,
                "fcpxml_path": fcpxml_path,
                "fcpxml_rel": fcpxml_rel,
                "file_bytes": file_bytes,
                "report": rep_obj,
                "report_json": report_json,
                "state": _build_project_state_payload(project),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def register_shotvault_prompt_id(prompt_id: Optional[str]) -> None:
    if prompt_id:
        with _LOCK:
            pid = str(prompt_id)
            if pid not in _SHOTVAULT_PROMPT_IDS:
                _SHOTVAULT_PROMPT_IDS.append(pid)


def register_current_running_prompt() -> None:
    try:
        from server import PromptServer  # type: ignore

        server = getattr(PromptServer, "instance", None)
        pid = getattr(server, "last_prompt_id", None)
        if pid:
            register_shotvault_prompt_id(str(pid))
    except Exception:
        pass


def is_shotvault_prompt_id(prompt_id: Optional[str]) -> bool:
    if not prompt_id:
        return False
    with _LOCK:
        return str(prompt_id) in _SHOTVAULT_PROMPT_IDS


def reset_abort_flag() -> None:
    global _ABORT_AUTO_QUEUE
    with _LOCK:
        _ABORT_AUTO_QUEUE = False


def trigger_abort_flag() -> None:
    global _ABORT_AUTO_QUEUE
    with _LOCK:
        _ABORT_AUTO_QUEUE = True


def is_aborted() -> bool:
    with _LOCK:
        return _ABORT_AUTO_QUEUE


def purge_auto_queued_prompts() -> int:
    """Remove pending ShotVault-spawned items from `PromptServer.instance.prompt_queue` on cancel."""
    try:
        from server import PromptServer  # type: ignore
    except ImportError:
        return 0
    server = getattr(PromptServer, "instance", None)
    queue = getattr(server, "prompt_queue", None)
    if queue is None:
        return 0
    removed = 0
    with _LOCK:
        tracked = set(_SHOTVAULT_PROMPT_IDS)
    if not tracked:
        return 0
    try:
        with queue.mutex:
            remaining = []
            for item in queue.queue:
                if len(item) > 1 and str(item[1]) in tracked:
                    removed += 1
                else:
                    remaining.append(item)
            if removed:
                import heapq

                queue.queue = remaining
                heapq.heapify(queue.queue)
                server.queue_updated()
    except Exception as exc:
        LOGGER.debug("ShotVault queue purge skipped: %s", exc)
    return removed


def handle_cancel_request(payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Trip abort latch only for forced user clicks or errors from a tracked ShotVault prompt."""
    data = payload or {}
    pid = str(data.get("prompt_id") or "")
    force = bool(data.get("force", False))
    with _LOCK:
        has_tracked = bool(_SHOTVAULT_PROMPT_IDS)
    should_abort = force or is_shotvault_prompt_id(pid) or (not pid and not has_tracked)
    purged = 0
    if should_abort:
        trigger_abort_flag()
        purged = purge_auto_queued_prompts()
    return {"aborted": is_aborted(), "matched": should_abort, "purged": purged}


def _on_user_prompt(json_data: dict[str, Any]) -> dict[str, Any]:
    """PromptServer `on_prompt` handler: reset `_ABORT_AUTO_QUEUE` whenever a ShotVault graph is queued."""
    if isinstance(json_data, dict):
        prompt = json_data.get("prompt")
        if isinstance(prompt, dict):
            has_sv = any(
                isinstance(n, dict) and str(n.get("class_type") or "").startswith("ShotVault")
                for n in prompt.values()
            )
            if has_sv:
                reset_abort_flag()
                pid = json_data.get("prompt_id")
                if pid:
                    register_shotvault_prompt_id(str(pid))
    return json_data


def schedule_next_queue_head(remaining_shots: list[int]) -> bool:
    """Clone currently running graph and enqueue next remaining shots at front (`number = -server.number`)."""
    if not remaining_shots:
        return False
    try:
        from server import PromptServer  # type: ignore
    except ImportError:
        return False

    server = getattr(PromptServer, "instance", None)
    queue = getattr(server, "prompt_queue", None)
    prompt_id = getattr(server, "last_prompt_id", None)
    if queue is None or not prompt_id:
        return False

    with queue.mutex:
        running = next((item for item in queue.currently_running.values() if item[1] == prompt_id), None)
        running = copy.deepcopy(running) if running is not None else None
    if running is None:
        return False

    prompt = running[2]
    next_spec = ",".join(str(int(s)) for s in remaining_shots)
    updated_any = False
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "ShotVaultLoad":
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs.get("shot_numbers"), list):
            inputs["shot_numbers"] = next_spec
            updated_any = True
    if not updated_any:
        return False

    new_pid = str(uuid.uuid4())
    register_shotvault_prompt_id(new_pid)
    number = -int(server.number)
    server.number += 1
    queue.put((number, new_pid, prompt, *running[3:]))
    LOGGER.info("ShotVault: queued remaining shots [%s] at queue head (#%d).", next_spec, number)
    return True


def _get_input_root() -> str:
    try:
        import folder_paths  # type: ignore

        return os.path.abspath(folder_paths.get_input_directory())
    except Exception:
        return os.path.abspath(os.path.join(os.getcwd(), "input"))


def _probe_media_meta(file_path: str) -> tuple[float, int, int]:
    """Return `(duration_sec, width, height)` for audio/video/image uploads."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        try:
            from PIL import Image  # type: ignore

            with Image.open(file_path) as im:
                return 0.0, int(im.width), int(im.height)
        except Exception:
            return 0.0, 0, 0
    try:
        import av  # type: ignore

        with av.open(file_path) as container:
            dur = float(container.duration / 1_000_000.0) if container.duration else 0.0
            v = next((s for s in container.streams if s.type == "video"), None)
            if v is not None:
                if dur <= 0.0 and v.frames and v.average_rate:
                    dur = float(v.frames) / float(v.average_rate)
                return round(max(0.0, dur), 3), int(v.width or 0), int(v.height or 0)
            a = next((s for s in container.streams if s.type == "audio"), None)
            if a is not None and dur <= 0.0 and a.duration and a.time_base:
                dur = float(a.duration * a.time_base)
            return round(max(0.0, dur), 3), 0, 0
    except Exception:
        return 0.0, 0, 0


def handle_upload_bytes(raw_bytes: bytes, filename: str, subfolder: str = "shotvault", input_root: Optional[str] = None) -> dict[str, Any]:
    """Save uploaded media bytes into `<input>/<subfolder>/` safely and probe duration/dimensions."""
    base_input = os.path.abspath(input_root or _get_input_root())
    safe_sub = sanitize_project_name(subfolder or "shotvault", fallback="shotvault")
    dest_dir = ensure_inside_root(base_input, safe_sub)
    os.makedirs(dest_dir, exist_ok=True)

    raw_name = os.path.basename(str(filename or "upload.bin"))
    stem, ext = os.path.splitext(raw_name)
    safe_stem = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".") or ord(ch) > 127) else "_" for ch in stem).strip(" ._") or "media"
    safe_ext = "".join(ch for ch in ext.lower() if ch.isalnum() or ch == ".") or ".bin"
    candidate = f"{safe_stem}{safe_ext}"
    idx = 1
    target_path = ensure_inside_root(dest_dir, candidate)
    while os.path.exists(target_path):
        candidate = f"{safe_stem}_{idx}{safe_ext}"
        target_path = ensure_inside_root(dest_dir, candidate)
        idx += 1

    with open(target_path, "wb") as f:
        f.write(raw_bytes)
    dur, w, h = _probe_media_meta(target_path)
    rel_path = f"{safe_sub}/{candidate}"
    return {"ok": True, "rel_path": rel_path, "filename": candidate, "duration_sec": dur, "width": w, "height": h}


def resolve_input_media_path(rel_file: str, input_root: Optional[str] = None) -> str:
    """Resolve `file` strictly inside `input_root`, raising `ValueError` on traversal."""
    base_input = os.path.abspath(input_root or _get_input_root())
    clean = str(rel_file or "").replace("\\", "/")
    if clean.startswith("input/"):
        clean = clean[6:]
    return ensure_inside_root(base_input, clean)


def handle_parse_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse markdown or dual-JSONC text into visual Director Console shot cards."""
    shots = parse_script(
        script_text=str(payload.get("script_text") or ""),
        common_prompt=str(payload.get("common_prompt") or ""),
        stage3_json_text=str(payload.get("stage3_json_text") or ""),
        script_format=str(payload.get("script_format") or "auto"),
        fps=int(payload.get("fps") or 24),
        has_seed_video=bool(payload.get("seed_video_path")),
        default_duration_sec=float(payload.get("default_duration_sec") or 5.0),
    )
    cards = [
        {
            "shot_index": s["shot_index"],
            "duration_sec": float(s["duration_sec"]),
            "shot_type": int(s["shot_type"]),
            "hard_cut": not bool(s["is_continuation"]),
            "overlap_override": int(s.get("overlap_override") or 0),
            "segment_only": bool(s.get("segment_only", False)),
            "inherit_global_refs": True,
            "inherit_global_audios": True,
            "prompt": s["local_prompt"],
            "ref_images": list(s.get("ref_images") or ()),
            "ref_slots": list(s.get("ref_slots") or s.get("ref_images") or ()),
            "ref_videos": list(s.get("ref_videos") or ()),
            "ref_audios": list(s.get("ref_audios") or ()),
            "ref_audio_slots": list(s.get("ref_audio_slots") or (list(s.get("ref_audios") or ()) + ["", "", ""])[:3]),
            "guide_audio": str(s.get("audio_file") or ""),
        }
        for s in shots
    ]
    return {"ok": True, "shots": cards}


def handle_sync_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist Director UI cards directly into `project.json` so `/shotvault/state` shows UNRENDERED shots live."""
    project_name = str(payload.get("project") or "MyH3Film")
    root = resolve_output_root(project_name)
    store = VaultStore(root)
    proj = store.load_project()
    fps = int(payload.get("fps") or proj.get("fps") or 24)
    common_prompt = str(payload.get("common_prompt") or "")
    gcfg, shots = parse_director_plan_json(
        str(payload.get("director_plan_json") or ""),
        common_prompt=common_prompt,
        fps=fps,
        has_seed_video=bool(proj.get("seed_shot")),
    )
    if gcfg.get("context_frames") in (22, 39, 56):
        proj["context_frames"] = int(gcfg["context_frames"])
    if shots:
        proj["last_plan"] = [
            {
                "shot_index": s["shot_index"],
                "shot_id": s["shot_id"],
                "duration_sec": s["duration_sec"],
                "shot_type": s["shot_type"],
                "is_continuation": s["is_continuation"],
                "overlap_override": int(s.get("overlap_override") or 0),
                "prompt_sha256": s["prompt_sha256"],
                "local_prompt": s["local_prompt"],
            }
            for s in shots
        ]
        proj["planned_shots"] = proj["last_plan"]
        store.save_project(proj)
    return _build_project_state_payload(project_name)


def _build_project_state_payload(project_name: str) -> dict[str, Any]:
    root = resolve_output_root(project_name)
    store = VaultStore(root)
    proj = store.load_project()
    last_plan = list(proj.get("last_plan") or proj.get("planned_shots") or ())
    if not last_plan and proj.get("shots"):
        last_plan = [
            {
                "shot_index": int(k),
                "shot_id": f"shot_{int(k):04d}",
                "duration_sec": 5.0,
                "shot_type": 1,
                "is_continuation": int(k) > 1,
                "prompt_sha256": "",
                "local_prompt": "",
            }
            for k in sorted(proj["shots"].keys(), key=lambda x: int(x))
        ]
    dag = LineageDAG.evaluate(last_plan, proj)
    shots_out = []
    for plan_item, dag_item in zip(last_plan, dag):
        idx_str = str(plan_item["shot_index"])
        rec = (proj.get("shots") or {}).get(idx_str) or {}
        takes_list = []
        for tid, tmeta in sorted((rec.get("takes") or {}).items()):
            takes_list.append({
                "take_id": tid,
                "seed_text": tmeta.get("seed_text", ""),
                "parent_take_id": tmeta.get("parent_take_id", ""),
                "keep_frames": tmeta.get("keep_frames", 0),
                "width": tmeta.get("width", 0),
                "height": tmeta.get("height", 0),
                "thumb_rel": tmeta.get("thumb_rel_path", ""),
                "seam_rel": tmeta.get("seam_rel_path", ""),
                "video_rel": tmeta.get("video_rel_path", ""),
                "seam_diagnostics": tmeta.get("seam_diagnostics") or {},
                "bridged_next_take_id": tmeta.get("bridged_next_take_id", ""),
                "tail_bridge_frames": int(tmeta.get("tail_bridge_frames") or 0),
            })
        shots_out.append({**plan_item, **dag_item, "takes": takes_list, "seam_repair": rec.get("seam_repair") or {}})
    return {
        "project_name": proj.get("project_name", project_name),
        "project_root": root,
        "seed_shot": proj.get("seed_shot"),
        "shots": shots_out,
        "master_exports": _scan_master_exports(root),
        "master_sync_status": compute_master_sync_status(store, last_plan, proj),
    }


def register_routes() -> None:
    """Register `/shotvault/*` REST endpoints and `on_prompt` reset handler on `PromptServer.instance`."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web  # type: ignore
        from server import PromptServer  # type: ignore
    except ImportError:
        return

    server = getattr(PromptServer, "instance", None)
    routes = getattr(server, "routes", None)
    if routes is None:
        return

    if hasattr(server, "add_on_prompt_handler"):
        server.add_on_prompt_handler(_on_user_prompt)

    @routes.get("/shotvault/projects")
    async def list_projects(_request):
        base_parent = os.path.dirname(resolve_output_root("_probe"))
        items = []
        if os.path.isdir(base_parent):
            for fn in sorted(os.listdir(base_parent)):
                if os.path.isfile(os.path.join(base_parent, fn, "project.json")):
                    items.append(fn)
        return web.json_response({"projects": items})

    @routes.get("/shotvault/state")
    async def get_state(request):
        project = request.rel_url.query.get("project", "MyH3Film")
        return web.json_response(_build_project_state_payload(project))

    @routes.get("/shotvault/media")
    async def get_media(request):
        project = request.rel_url.query.get("project", "MyH3Film")
        rel_file = request.rel_url.query.get("file", "")
        download = request.rel_url.query.get("download", "") in ("1", "true", "yes")
        root = resolve_output_root(project)
        try:
            target = ensure_inside_root(root, rel_file)
        except ValueError:
            return web.Response(status=403, text="Forbidden path")
        if not os.path.isfile(target):
            return web.Response(status=404, text="Not found")
        headers = None
        if download:
            fname = os.path.basename(target)
            headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"}
        return web.FileResponse(target, headers=headers)

    @routes.get("/shotvault/input-media")
    async def get_input_media(request):
        rel_file = request.rel_url.query.get("file", "")
        try:
            target = resolve_input_media_path(rel_file)
        except ValueError:
            return web.Response(status=403, text="Forbidden path")
        if not os.path.isfile(target):
            return web.Response(status=404, text="Not found")
        return web.FileResponse(target)

    @routes.post("/shotvault/upload")
    async def upload_media(request):
        post = await request.post()
        field = post.get("file") or post.get("image")
        if field is None or not hasattr(field, "file"):
            return web.json_response({"ok": False, "error": "Missing multipart file field"}, status=400)
        subfolder = str(post.get("subfolder") or "shotvault")
        raw_bytes = field.file.read()
        return web.json_response(handle_upload_bytes(raw_bytes, getattr(field, "filename", "upload.bin"), subfolder=subfolder))

    @routes.post("/shotvault/parse-preview")
    async def parse_preview(request):
        payload = await request.json()
        return web.json_response(handle_parse_preview(payload))

    @routes.post("/shotvault/sync-plan")
    async def sync_plan(request):
        payload = await request.json()
        return web.json_response(handle_sync_plan(payload))

    @routes.post("/shotvault/select-take")
    async def select_take(request):
        data = await request.json()
        store = VaultStore(resolve_output_root(data.get("project", "MyH3Film")))
        store.set_active_take(int(data["shot_index"]), str(data["take_id"]))
        return web.json_response(_build_project_state_payload(data.get("project", "MyH3Film")))

    @routes.post("/shotvault/lock-shot")
    async def lock_shot(request):
        data = await request.json()
        store = VaultStore(resolve_output_root(data.get("project", "MyH3Film")))
        store.set_shot_locked(int(data["shot_index"]), bool(data.get("locked", True)))
        return web.json_response(_build_project_state_payload(data.get("project", "MyH3Film")))

    @routes.post("/shotvault/cleanup-recycle")
    async def cleanup_recycle(request):
        data = await request.json()
        store = VaultStore(resolve_output_root(data.get("project", "MyH3Film")))
        return web.json_response(store.cleanup_unused_takes())

    @routes.get("/shotvault/history-inventory")
    async def history_inventory(request):
        project = request.rel_url.query.get("project", "MyH3Film")
        store = VaultStore(resolve_output_root(project))
        return web.json_response(store.get_history_inventory())

    @routes.post("/shotvault/cleanup-items")
    async def cleanup_items_route(request):
        try:
            data = await request.json()
            project = data.get("project", "MyH3Film")
            store = VaultStore(resolve_output_root(project))
            res = store.cleanup_items(
                take_ids=data.get("take_ids"),
                scope=str(data.get("scope") or "selected"),
                shot_index=int(data["shot_index"]) if data.get("shot_index") is not None else None,
                mode=str(data.get("mode") or "recycle"),
                allow_protected=bool(data.get("allow_protected", False)),
            )
            res["state"] = _build_project_state_payload(project)
            res["inventory"] = store.get_history_inventory()
            return web.json_response(res)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/shotvault/recycle-bin")
    async def recycle_bin_route(request):
        try:
            data = await request.json()
            project = data.get("project", "MyH3Film")
            store = VaultStore(resolve_output_root(project))
            res = store.manage_recycle_bin(
                action=str(data.get("action") or "list"),
                batch_ids=data.get("batch_ids"),
            )
            res["state"] = _build_project_state_payload(project)
            res["inventory"] = store.get_history_inventory()
            return web.json_response(res)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/shotvault/repair-seam")
    async def repair_seam_route(request):
        from .core.seam_repair_bridge import repair_shot_seam_preview

        data = await request.json()
        project = data.get("project", "MyH3Film")
        store = VaultStore(resolve_output_root(project))
        try:
            res = repair_shot_seam_preview(
                store=store,
                shot_index=int(data["shot_index"]),
                take_id=data.get("take_id"),
                mode=str(data.get("mode") or "rife"),
                rife_bridge_frames=int(data.get("rife_bridge_frames") or 5),
                rife_backend=str(data.get("rife_backend") or "auto"),
                preserve_duration=bool(data.get("preserve_duration", False)),
                max_search_sec=float(data.get("max_search_sec", 0.5)),
                structure_threshold=float(data.get("structure_threshold", 0.38)),
                coverage_threshold=float(data.get("coverage_threshold", 0.08)),
                release_seconds=float(data.get("release_seconds", 0.8)),
                retry_policy=str(data.get("retry_policy") or "off"),
            )
            res["state"] = _build_project_state_payload(project)
            return web.json_response(res)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/shotvault/set-seam-choice")
    async def set_seam_choice_route(request):
        try:
            data = await request.json()
            project = data.get("project", "MyH3Film")
            shot_idx = int(data["shot_index"])
            store = VaultStore(resolve_output_root(project))
            patch: dict[str, Any] = {}
            for k in ("enabled", "choice", "mode", "rife_bridge_frames", "preserve_duration", "structure_threshold", "coverage_threshold", "retry_policy"):
                if k in data:
                    patch[k] = data[k]
            sr = store.set_shot_seam_repair(shot_idx, patch)
            return web.json_response({"ok": True, "seam_repair": sr, "state": _build_project_state_payload(project)})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/shotvault/cancel-auto")
    async def cancel_auto(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        return web.json_response(handle_cancel_request(payload))

    @routes.post("/shotvault/resume-auto")
    async def resume_auto(_request):
        reset_abort_flag()
        return web.json_response({"aborted": False})

    @routes.post("/shotvault/assemble")
    async def api_assemble(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        res = await asyncio.to_thread(handle_assemble_request, data)
        return web.json_response(res)

    _ROUTES_REGISTERED = True
    LOGGER.info("ShotVault: registered /shotvault/* HTTP API routes and on_prompt handler.")
