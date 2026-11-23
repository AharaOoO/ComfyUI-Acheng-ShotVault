"""Persistent ShotVault project store: atomic project.json, .reserve locks, .h3ctx.safetensors."""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import uuid
from typing import Any, Optional

_PROJECT_IO_LOCK = threading.Lock()

import torch
import torch.nn.functional as F
from safetensors.torch import load as safetensors_load_bytes
from safetensors.torch import save_file as safetensors_save_file

from ..compat import ensure_inside_root, resample_audio, safe_recycle
from .grid_math import AUDIO_VAE_SR, DELIVERY_AUDIO_SR, FPS_DEFAULT, compute_audio_overhang, video_latent_t


class VaultStore:
    """Manage `output/ShotVault/<project_name>/` storage, atomic take reservations, and `.h3ctx` files."""

    SUBDIRS = (
        "script",
        os.path.join("assets", "chars"),
        os.path.join("assets", "props"),
        os.path.join("assets", "scenes"),
        os.path.join("assets", "voices"),
        "master",
        "shots",
        "_vault",
        os.path.join("_vault", "_recycle_bin"),
    )

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(self.project_root, sub), exist_ok=True)
        self.project_json_path = os.path.join(self.project_root, "project.json")
        self.vault_dir = os.path.join(self.project_root, "_vault")
        self.shots_dir = os.path.join(self.project_root, "shots")
        self.master_dir = os.path.join(self.project_root, "master")
        self.script_dir = os.path.join(self.project_root, "script")
        self.recycle_dir = os.path.join(self.vault_dir, "_recycle_bin")

    def load_project(self) -> dict[str, Any]:
        """Load `project.json` or return default schema."""
        if not os.path.isfile(self.project_json_path):
            return {
                "schema_version": "4.0",
                "project_name": os.path.basename(self.project_root),
                "seed_shot": None,
                "shots": {},
                "queue_cursor": None,
                "last_plan": [],
            }
        with open(self.project_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("schema_version", "4.0")
        data.setdefault("shots", {})
        return data

    def save_project(self, data: dict[str, Any]) -> None:
        """Atomically persist `project.json` using thread lock + unique temp file + `os.replace`."""
        with _PROJECT_IO_LOCK:
            tmp = f"{self.project_json_path}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.project_json_path)

    def reserve_next_take(self, shot_index: int) -> tuple[str, str]:
        """Atomically reserve the next unused `shot_{idx:04d}_take_{n:03d}` ID via `.reserve` O_EXCL lock."""
        prefix = f"shot_{int(shot_index):04d}_take_"
        max_n = 0
        for folder in (self.shots_dir, self.vault_dir):
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if not fn.startswith(prefix):
                    continue
                tail = fn[len(prefix):]
                num_str = tail.split(".", 1)[0].split("_", 1)[0]
                if num_str.isdigit():
                    max_n = max(max_n, int(num_str))
        # Also scan recycle-bin manifests so restoring a deleted take never collides with a newly rendered take
        if os.path.isdir(self.recycle_dir):
            for bname in os.listdir(self.recycle_dir):
                mpath = os.path.join(self.recycle_dir, bname, "manifest.json")
                if not os.path.isfile(mpath):
                    continue
                try:
                    with open(mpath, "r", encoding="utf-8") as mf:
                        mobj = json.load(mf)
                    for tid in mobj.get("take_ids") or ():
                        if str(tid).startswith(prefix):
                            ns = str(tid)[len(prefix):].split("_", 1)[0]
                            if ns.isdigit():
                                max_n = max(max_n, int(ns))
                except Exception:
                    pass

        candidate_n = max_n + 1
        while True:
            take_id = f"{prefix}{candidate_n:03d}"
            lock_path = os.path.join(self.vault_dir, f"{take_id}.reserve")
            ctx_path = os.path.join(self.vault_dir, f"{take_id}.h3ctx.safetensors")
            mp4_path = os.path.join(self.shots_dir, f"{take_id}.mp4")
            if os.path.exists(ctx_path) or os.path.exists(mp4_path):
                candidate_n += 1
                continue
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"{os.getpid()}\n".encode("ascii"))
                finally:
                    os.close(fd)
                return take_id, lock_path
            except FileExistsError:
                candidate_n += 1

    @staticmethod
    def release_reserve(lock_path: str) -> None:
        if lock_path and os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except OSError:
                pass

    @staticmethod
    def _resize_frames_chw(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if frames.shape[1] == target_h and frames.shape[2] == target_w:
            return frames
        chw = frames[..., :3].float().movedim(-1, 1)
        resized = F.interpolate(chw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized.movedim(1, -1)

    def save_take_context(
        self,
        take_id: str,
        shot_index: int,
        clean_images: torch.Tensor,
        audio_waveform_48k: Optional[torch.Tensor],
        audio_sr: int,
        fps: int,
        prompt_sha256: str,
        parent_take_id: str,
        seed_text: str,
        keep_frames: int,
        trim_head_frames: int,
        shot_type: int,
        parent_clean_tail: Optional[torch.Tensor] = None,
        sampled_latent: Optional[dict] = None,
        tail_latent_frames: int = 39,
        max_clean_tail_frames: int = 56,
        video_rel_path: str = "",
        thumb_rel_path: str = "",
        seam_rel_path: str = "",
        seam_diagnostics: Optional[dict[str, Any]] = None,
        set_active: bool = True,
        raw_trimmed_frames: Optional[int] = None,
    ) -> dict[str, Any]:
        """Save clean context tail, bare 12-step AV latent tail (when tail is untrimmed), 32kHz & 48kHz audio, and metadata."""
        raw_count = int(raw_trimmed_frames) if raw_trimmed_frames is not None else int(clean_images.shape[0])
        clean_cpu = clean_images[..., :3].detach().cpu().float().clamp(0.0, 1.0)
        if clean_cpu.shape[0] < max_clean_tail_frames and parent_clean_tail is not None and parent_clean_tail.shape[0] > 0:
            p_tail = parent_clean_tail[..., :3].detach().cpu().float().clamp(0.0, 1.0)
            p_tail = self._resize_frames_chw(p_tail, int(clean_cpu.shape[1]), int(clean_cpu.shape[2]))
            need = max_clean_tail_frames - int(clean_cpu.shape[0])
            clean_cpu = torch.cat([p_tail[-need:], clean_cpu], dim=0)
        clean_tail = clean_cpu[-max_clean_tail_frames:].to(torch.float16).contiguous()

        if audio_waveform_48k is not None and audio_waveform_48k.numel() > 0:
            wav = audio_waveform_48k.detach().cpu().float()
            if wav.ndim == 3:
                wav = wav[0]
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]
            wav_48k = resample_audio(wav, int(audio_sr), DELIVERY_AUDIO_SR).float().contiguous()
            wav_32k_full = resample_audio(wav, int(audio_sr), AUDIO_VAE_SR).float().contiguous()
        else:
            samples_48 = max(1, round(int(clean_images.shape[0]) / float(fps) * DELIVERY_AUDIO_SR))
            samples_32 = max(1, round(int(clean_tail.shape[0]) / float(fps) * AUDIO_VAE_SR))
            wav_48k = torch.zeros((2, samples_48), dtype=torch.float32)
            wav_32k_full = torch.zeros((2, samples_32), dtype=torch.float32)

        # Store flush end-aligned 32kHz audio tail (>= 96 * 800 samples); left-pad if shot is shorter so -1 stays flush with seam.
        tail_audio_samples = max(96 * 800, round(max(56, int(clean_tail.shape[0])) / float(fps) * AUDIO_VAE_SR))
        if wav_32k_full.shape[1] < tail_audio_samples:
            audio_32k_tail = F.pad(wav_32k_full, (tail_audio_samples - int(wav_32k_full.shape[1]), 0)).contiguous()
        else:
            audio_32k_tail = wav_32k_full[:, -tail_audio_samples:].contiguous()

        tensors: dict[str, torch.Tensor] = {
            "clean_tail": clean_tail,
            "audio_32k_tail": audio_32k_tail,
            "delivery_audio_48k": wav_48k,
        }

        overhang = 0.0
        has_tail_latent = False
        tail_latent_skipped_reason = ""
        if isinstance(sampled_latent, dict) and "samples" in sampled_latent:
            samples = sampled_latent["samples"]
            v_lat = a_lat = None
            if getattr(samples, "is_nested", False) or hasattr(samples, "unbind"):
                parts = list(samples.unbind())
                if len(parts) >= 2:
                    v_lat, a_lat = parts[0], parts[1]
            elif isinstance(samples, (list, tuple)) and len(samples) >= 2:
                v_lat, a_lat = samples[0], samples[1]
            if isinstance(v_lat, torch.Tensor) and v_lat.ndim == 5 and v_lat.shape[2] >= 2:
                req_steps = video_latent_t(tail_latent_frames)
                if v_lat.shape[2] < req_steps:
                    req_steps = video_latent_t(22)
                if v_lat.shape[2] >= req_steps and (v_lat.shape[2] - 2) % 5 == 0:
                    gen_frames = 5 + 17 * ((int(v_lat.shape[2]) - 2) // 5)
                    used_end_frame = int(trim_head_frames) + raw_count
                    if used_end_frame == gen_frames:
                        tensors["tail_latent_video"] = v_lat[:1, :, -req_steps:].detach().cpu().float().contiguous()
                        if isinstance(a_lat, torch.Tensor) and a_lat.shape[-1] > 0:
                            req_a = max(1, round(tail_latent_frames * (40.0 / float(fps))))
                            tensors["tail_latent_audio"] = a_lat[:1, ..., -req_a:].detach().cpu().float().contiguous()
                            overhang = compute_audio_overhang(int(a_lat.shape[-1]), gen_frames)
                        has_tail_latent = True
                    else:
                        tail_latent_skipped_reason = "tail_trimmed"

        meta = {
            "take_id": take_id,
            "shot_index": int(shot_index),
            "prompt_sha256": str(prompt_sha256),
            "parent_take_id": str(parent_take_id or ""),
            "seed_text": str(seed_text),
            "keep_frames": int(keep_frames),
            "trim_head_frames": int(trim_head_frames),
            "shot_type": int(shot_type),
            "fps": int(fps),
            "clean_tail_frames": int(clean_tail.shape[0]),
            "width": int(clean_tail.shape[2]),
            "height": int(clean_tail.shape[1]),
            "has_tail_latent": has_tail_latent,
            "tail_latent_skipped_reason": tail_latent_skipped_reason,
            "audio_overhang": float(overhang),
            "video_rel_path": video_rel_path or f"shots/{take_id}.mp4",
            "thumb_rel_path": thumb_rel_path or f"_vault/thumb_{take_id}.jpg",
            "seam_rel_path": seam_rel_path or "",
            "seam_diagnostics": seam_diagnostics or {},
        }

        ctx_path = os.path.join(self.vault_dir, f"{take_id}.h3ctx.safetensors")
        safetensors_save_file(tensors, ctx_path, metadata={"shotvault_meta": json.dumps(meta, ensure_ascii=False)})

        meta_path = os.path.join(self.vault_dir, f"{take_id}.meta.json")
        tmp_meta = f"{meta_path}.tmp.{os.getpid()}"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, meta_path)

        project = self.load_project()
        shot_rec = project["shots"].setdefault(
            str(shot_index),
            {"active_take_id": "", "locked_keep": False, "takes": {}},
        )
        shot_rec["takes"][take_id] = meta
        if set_active or not shot_rec.get("active_take_id"):
            shot_rec["active_take_id"] = take_id
        self.save_project(project)
        return meta

    def load_take_context(self, take_id: str) -> Optional[dict[str, Any]]:
        """Load `.h3ctx.safetensors` tensors and metadata for `take_id`."""
        if not take_id:
            return None
        safe_id = os.path.basename(take_id)
        ctx_path = ensure_inside_root(self.vault_dir, f"{safe_id}.h3ctx.safetensors")
        if not os.path.isfile(ctx_path):
            return None
        with open(ctx_path, "rb") as f:
            raw = f.read()
        tensors = safetensors_load_bytes(raw)
        header_meta: dict[str, Any] = {}
        if len(raw) >= 8:
            hdr_len = struct.unpack("<Q", raw[:8])[0]
            if 8 + hdr_len <= len(raw):
                hdr_obj = json.loads(raw[8:8 + hdr_len].decode("utf-8"))
                raw_meta = (hdr_obj.get("__metadata__") or {}).get("shotvault_meta", "")
                if raw_meta:
                    header_meta = json.loads(raw_meta)
        meta_path = ensure_inside_root(self.vault_dir, f"{safe_id}.meta.json")
        if not header_meta and os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                header_meta = json.load(f)
        return {"tensors": tensors, "meta": header_meta}

    def get_active_take_id(self, shot_index: int) -> str:
        project = self.load_project()
        if int(shot_index) == 0:
            seed = project.get("seed_shot") or {}
            return str(seed.get("take_id", ""))
        rec = (project.get("shots") or {}).get(str(int(shot_index)), {})
        return str(rec.get("active_take_id", ""))

    def set_active_take(self, shot_index: int, take_id: str) -> dict[str, Any]:
        project = self.load_project()
        key = str(int(shot_index))
        rec = project["shots"].get(key)
        if not rec or take_id not in (rec.get("takes") or {}):
            raise KeyError(f"ShotVault: take {take_id!r} not found for shot {shot_index}.")
        rec["active_take_id"] = take_id
        self.save_project(project)
        return project

    def set_shot_locked(self, shot_index: int, locked: bool) -> dict[str, Any]:
        project = self.load_project()
        key = str(int(shot_index))
        rec = project["shots"].setdefault(key, {"active_take_id": "", "locked_keep": False, "takes": {}})
        rec["locked_keep"] = bool(locked)
        self.save_project(project)
        return project

    def set_shot_seam_repair(self, shot_index: int, patch: dict[str, Any]) -> dict[str, Any]:
        """Persist per-shot FuzzPuppy seam repair toggle, choice ('repaired'|'original'), and settings."""
        project = self.load_project()
        key = str(int(shot_index))
        rec = project["shots"].setdefault(key, {"active_take_id": "", "locked_keep": False, "takes": {}})
        cur = dict(rec.get("seam_repair") or {})
        cur.update(patch)
        rec["seam_repair"] = cur
        self.save_project(project)
        return cur

    def _compute_protected_reasons(self, project: dict[str, Any]) -> dict[str, list[str]]:
        reasons: dict[str, list[str]] = {}

        def _add(tid: str, why: str) -> None:
            if not tid:
                return
            lst = reasons.setdefault(str(tid), [])
            if why not in lst:
                lst.append(why)

        _add("shot_0000_seed", "seed")
        seed_shot = project.get("seed_shot") or {}
        if seed_shot.get("take_id"):
            _add(str(seed_shot["take_id"]), "seed")

        for rec in (project.get("shots") or {}).values():
            active = str(rec.get("active_take_id") or "")
            if active:
                _add(active, "active")
                active_meta = (rec.get("takes") or {}).get(active) or {}
                parent_tid = str(active_meta.get("parent_take_id") or "")
                if parent_tid:
                    _add(parent_tid, "parent_ref")
            acc_parent = str(rec.get("accepted_parent_take_id") or "")
            if acc_parent:
                _add(acc_parent, "parent_ref")
            if rec.get("locked_keep"):
                for tid in (rec.get("takes") or {}):
                    _add(tid, "locked")
        return reasons

    def _collect_take_rel_files(self, take_id: str, take_meta: Optional[dict[str, Any]] = None) -> list[str]:
        safe_id = os.path.basename(str(take_id))
        cands = [
            f"shots/{safe_id}.mp4",
            f"shots/{safe_id}.mov",
            f"shots/{safe_id}.txt",
            f"_vault/{safe_id}.h3ctx.safetensors",
            f"_vault/{safe_id}.meta.json",
            f"_vault/thumb_{safe_id}.jpg",
            f"_vault/seam_{safe_id}.mp4",
            f"_vault/seam_repaired_{safe_id}.mp4",
        ]
        if isinstance(take_meta, dict):
            for k in ("video_rel_path", "thumb_rel_path", "seam_rel_path"):
                rel = str(take_meta.get(k) or "").replace("\\", "/")
                if rel and rel not in cands:
                    cands.append(rel)
        existing: list[str] = []
        for rel in cands:
            try:
                abs_p = ensure_inside_root(self.project_root, rel)
            except ValueError:
                continue
            if os.path.isfile(abs_p):
                existing.append(rel)
        return existing

    @staticmethod
    def _dir_size_bytes(folder: str) -> int:
        total = 0
        if not os.path.isdir(folder):
            return 0
        for r, _, fns in os.walk(folder):
            for fn in fns:
                try:
                    total += os.path.getsize(os.path.join(r, fn))
                except OSError:
                    pass
        return total

    def get_history_inventory(self) -> dict[str, Any]:
        """Return full per-take inventory, protection flags, cache sizes, and recycle-bin batches."""
        project = self.load_project()
        prot_map = self._compute_protected_reasons(project)
        shots_inv: list[dict[str, Any]] = []
        total_unused_bytes = 0
        total_unused_count = 0

        for k in sorted((project.get("shots") or {}).keys(), key=lambda x: int(x)):
            rec = project["shots"][k] or {}
            shot_idx = int(k)
            takes_out: list[dict[str, Any]] = []
            for tid, tmeta in sorted((rec.get("takes") or {}).items()):
                rels = self._collect_take_rel_files(tid, tmeta)
                size_b = 0
                cache_b = 0
                for rel in rels:
                    try:
                        sz = os.path.getsize(ensure_inside_root(self.project_root, rel))
                    except OSError:
                        sz = 0
                    size_b += sz
                    if os.path.basename(rel).startswith(("seam_", "seam_repaired_")):
                        cache_b += sz
                reasons = prot_map.get(tid, [])
                deletable = len(reasons) == 0
                if deletable:
                    total_unused_bytes += size_b
                    total_unused_count += 1
                takes_out.append({
                    "take_id": tid,
                    "shot_index": shot_idx,
                    "seed_text": str((tmeta or {}).get("seed_text") or ""),
                    "width": int((tmeta or {}).get("width") or 0),
                    "height": int((tmeta or {}).get("height") or 0),
                    "keep_frames": int((tmeta or {}).get("keep_frames") or 0),
                    "created_at": float((tmeta or {}).get("created_at") or 0.0),
                    "bytes": size_b,
                    "cache_bytes": cache_b,
                    "file_count": len(rels),
                    "is_active": "active" in reasons,
                    "is_locked": "locked" in reasons,
                    "is_parent_ref": "parent_ref" in reasons,
                    "protected_reasons": reasons,
                    "deletable": deletable,
                })
            shots_inv.append({
                "shot_index": shot_idx,
                "active_take_id": str(rec.get("active_take_id") or ""),
                "locked_keep": bool(rec.get("locked_keep", False)),
                "takes": takes_out,
            })

        # Repair cache summary
        cache_files = []
        cache_bytes = 0
        if os.path.isdir(self.vault_dir):
            for fn in sorted(os.listdir(self.vault_dir)):
                if fn.startswith("seam_repaired_") and fn.endswith(".mp4"):
                    fp = os.path.join(self.vault_dir, fn)
                    sz = os.path.getsize(fp) if os.path.isfile(fp) else 0
                    cache_files.append({"name": fn, "bytes": sz})
                    cache_bytes += sz

        # Recycle bin batches
        batches = self._list_recycle_batches()
        folders = {
            "shots": self._dir_size_bytes(self.shots_dir),
            "vault": self._dir_size_bytes(self.vault_dir),
            "master": self._dir_size_bytes(self.master_dir),
            "recycle": self._dir_size_bytes(self.recycle_dir),
        }
        return {
            "ok": True,
            "project_name": project.get("project_name", os.path.basename(self.project_root)),
            "shots": shots_inv,
            "unused_count": total_unused_count,
            "unused_bytes": total_unused_bytes,
            "repair_cache": {"count": len(cache_files), "bytes": cache_bytes, "files": cache_files},
            "recycle_bin": {"batches": batches, "total_bytes": folders["recycle"], "count": len(batches)},
            "folders": folders,
        }

    def _list_recycle_batches(self) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        if not os.path.isdir(self.recycle_dir):
            return batches
        for name in sorted(os.listdir(self.recycle_dir), reverse=True):
            bdir = os.path.join(self.recycle_dir, name)
            if os.path.isdir(bdir):
                mpath = os.path.join(bdir, "manifest.json")
                meta: dict[str, Any] = {}
                if os.path.isfile(mpath):
                    try:
                        with open(mpath, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                    except Exception:
                        meta = {}
                batches.append({
                    "batch_id": name,
                    "created_at": float(meta.get("created_at") or os.path.getmtime(bdir)),
                    "scope": str(meta.get("scope") or "legacy"),
                    "take_ids": list(meta.get("take_ids") or ()),
                    "file_count": len(meta.get("files") or ()),
                    "bytes": self._dir_size_bytes(bdir),
                    "restorable": bool(meta.get("files")),
                })
            elif os.path.isfile(bdir):
                batches.append({
                    "batch_id": name,
                    "created_at": float(os.path.getmtime(bdir)),
                    "scope": "legacy_file",
                    "take_ids": [],
                    "file_count": 1,
                    "bytes": os.path.getsize(bdir),
                    "restorable": False,
                })
        return batches

    def cleanup_items(
        self,
        take_ids: Optional[list[str]] = None,
        scope: str = "selected",
        shot_index: Optional[int] = None,
        mode: str = "recycle",
        allow_protected: bool = False,
    ) -> dict[str, Any]:
        """Clean single, multi-selected, shot-unused, project-unused takes, or repair preview caches cleanly."""
        import shutil
        import uuid

        project = self.load_project()
        prot_map = self._compute_protected_reasons(project)
        shots = project.get("shots") or {}

        # Cache-only cleanup mode: clears `seam_repaired_*.mp4` previews while keeping all takes intact
        if scope == "repair_cache_only":
            removed_files: list[dict[str, str]] = []
            cleared_seam_rels: list[dict[str, str]] = []
            perm_count = 0
            freed = 0
            batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            bdir = os.path.join(self.recycle_dir, batch_id)
            if os.path.isdir(self.vault_dir):
                for fn in sorted(os.listdir(self.vault_dir)):
                    if not (fn.startswith("seam_repaired_") and fn.endswith(".mp4")):
                        continue
                    rel = f"_vault/{fn}"
                    src = os.path.join(self.vault_dir, fn)
                    if not os.path.isfile(src):
                        continue
                    sz = os.path.getsize(src)
                    freed += sz
                    if mode == "permanent":
                        os.remove(src)
                        perm_count += 1
                    else:
                        os.makedirs(bdir, exist_ok=True)
                        dst = os.path.join(bdir, f"{len(removed_files):04d}_{fn}")
                        shutil.move(src, dst)
                        removed_files.append({"orig_rel": rel, "stored_name": os.path.basename(dst)})
            for sk, rec in shots.items():
                sr = rec.get("seam_repair")
                if isinstance(sr, dict) and sr.get("repaired_seam_rel"):
                    cleared_seam_rels.append({"shot_key": str(sk), "repaired_seam_rel": str(sr["repaired_seam_rel"])})
                    sr["repaired_seam_rel"] = ""
            self.save_project(project)
            if removed_files and mode != "permanent":
                with open(os.path.join(bdir, "manifest.json"), "w", encoding="utf-8") as mf:
                    json.dump({
                        "created_at": time.time(),
                        "scope": scope,
                        "take_ids": [],
                        "files": removed_files,
                        "removed_takes_meta": [],
                        "cleared_seam_rels": cleared_seam_rels,
                    }, mf, ensure_ascii=False, indent=2)
            return {
                "ok": True,
                "mode": mode,
                "scope": scope,
                "deleted_takes": [],
                "skipped_protected": [],
                "file_count": perm_count if mode == "permanent" else len(removed_files),
                "freed_bytes": freed,
                "batch_id": batch_id if removed_files and mode != "permanent" else "",
            }

        # Resolve target take IDs by scope
        target_pairs: list[tuple[str, str]] = []  # (shot_key, take_id)
        if scope == "shot_unused" and shot_index is not None:
            sk = str(int(shot_index))
            rec = shots.get(sk) or {}
            for tid in sorted((rec.get("takes") or {}).keys()):
                if tid not in prot_map:
                    target_pairs.append((sk, tid))
        elif scope == "project_unused":
            for sk in sorted(shots.keys(), key=lambda x: int(x)):
                rec = shots[sk] or {}
                for tid in sorted((rec.get("takes") or {}).keys()):
                    if tid not in prot_map:
                        target_pairs.append((sk, tid))
        else:
            want = {str(t) for t in (take_ids or ()) if t}
            for sk, rec in shots.items():
                for tid in (rec.get("takes") or {}).keys():
                    if tid in want:
                        target_pairs.append((sk, tid))

        deleted_takes: list[str] = []
        skipped_protected: list[dict[str, Any]] = []
        moved_files: list[dict[str, str]] = []
        removed_meta_records: list[dict[str, Any]] = []
        freed_bytes = 0
        file_count = 0

        first_tag = os.path.basename(target_pairs[0][1]) if target_pairs else "items"
        batch_id = f"batch_{int(time.time())}_{first_tag}_{uuid.uuid4().hex[:6]}"
        bdir = os.path.join(self.recycle_dir, batch_id)

        for sk, tid in target_pairs:
            reasons = prot_map.get(tid, [])
            if reasons and not allow_protected:
                skipped_protected.append({"take_id": tid, "reasons": reasons})
                continue
            # Even with allow_protected, never delete the currently active take directly
            rec = shots.get(sk) or {}
            if rec.get("active_take_id") == tid:
                skipped_protected.append({"take_id": tid, "reasons": ["active"]})
                continue
            tmeta = (rec.get("takes") or {}).get(tid) or {}
            rels = self._collect_take_rel_files(tid, tmeta)
            for rel in rels:
                src = ensure_inside_root(self.project_root, rel)
                if not os.path.isfile(src):
                    continue
                try:
                    freed_bytes += os.path.getsize(src)
                except OSError:
                    pass
                file_count += 1
                if mode == "permanent":
                    os.remove(src)
                else:
                    os.makedirs(bdir, exist_ok=True)
                    stored = f"{len(moved_files):04d}_{os.path.basename(rel)}"
                    shutil.move(src, os.path.join(bdir, stored))
                    moved_files.append({"orig_rel": rel, "stored_name": stored})

            removed_meta_records.append({"shot_key": sk, "take_id": tid, "meta": tmeta})
            (rec.get("takes") or {}).pop(tid, None)
            deleted_takes.append(tid)

        if deleted_takes:
            self.save_project(project)
            if mode != "permanent" and (moved_files or removed_meta_records):
                os.makedirs(bdir, exist_ok=True)
                with open(os.path.join(bdir, "manifest.json"), "w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "created_at": time.time(),
                            "scope": scope,
                            "take_ids": deleted_takes,
                            "files": moved_files,
                            "removed_takes_meta": removed_meta_records,
                        },
                        mf,
                        ensure_ascii=False,
                        indent=2,
                    )

        return {
            "ok": True,
            "mode": mode,
            "scope": scope,
            "deleted_takes": deleted_takes,
            "skipped_protected": skipped_protected,
            "file_count": file_count,
            "recycled_files": [os.path.basename(f["orig_rel"]) for f in moved_files],
            "freed_bytes": freed_bytes,
            "batch_id": batch_id if (deleted_takes and mode != "permanent") else "",
        }

    def manage_recycle_bin(self, action: str = "list", batch_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """List, restore (files + project.json metadata), or permanently empty `_recycle_bin` batches."""
        import shutil

        if action == "list":
            batches = self._list_recycle_batches()
            return {"ok": True, "batches": batches, "total_bytes": self._dir_size_bytes(self.recycle_dir)}

        if action == "restore":
            targets = [os.path.basename(b) for b in (batch_ids or ()) if b]
            restored_takes: list[str] = []
            restored_files = 0
            project = self.load_project()
            shots = project.setdefault("shots", {})
            for bid in targets:
                bdir = ensure_inside_root(self.recycle_dir, bid)
                mpath = os.path.join(bdir, "manifest.json")
                if not os.path.isdir(bdir) or not os.path.isfile(mpath):
                    continue
                with open(mpath, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                for fitem in manifest.get("files") or ():
                    src = os.path.join(bdir, str(fitem["stored_name"]))
                    dst = ensure_inside_root(self.project_root, str(fitem["orig_rel"]))
                    if os.path.isfile(src):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.move(src, dst)
                        restored_files += 1
                for trec in manifest.get("removed_takes_meta") or ():
                    sk = str(trec["shot_key"])
                    tid = str(trec["take_id"])
                    srec = shots.setdefault(sk, {"active_take_id": "", "locked_keep": False, "takes": {}})
                    srec.setdefault("takes", {})[tid] = trec.get("meta") or {}
                    restored_takes.append(tid)
                restored_cache = False
                for sref in manifest.get("cleared_seam_rels") or ():
                    sk = str(sref.get("shot_key") or "")
                    rel = str(sref.get("repaired_seam_rel") or "")
                    if sk and rel and sk in shots and isinstance(shots[sk].get("seam_repair"), dict):
                        shots[sk]["seam_repair"]["repaired_seam_rel"] = rel
                        restored_cache = True
                shutil.rmtree(bdir, ignore_errors=True)
            if restored_takes or restored_cache:
                self.save_project(project)
            return {"ok": True, "action": "restore", "restored_takes": restored_takes, "restored_files": restored_files}

        if action == "empty":
            targets = [os.path.basename(b) for b in (batch_ids or ()) if b]
            if not targets and os.path.isdir(self.recycle_dir):
                targets = sorted(os.listdir(self.recycle_dir))
            freed = 0
            removed = 0
            for bid in targets:
                target = ensure_inside_root(self.recycle_dir, bid)
                if os.path.isdir(target):
                    freed += self._dir_size_bytes(target)
                    shutil.rmtree(target, ignore_errors=True)
                    removed += 1
                elif os.path.isfile(target):
                    try:
                        freed += os.path.getsize(target)
                    except OSError:
                        pass
                    os.remove(target)
                    removed += 1
            return {"ok": True, "action": "empty", "removed_batches": removed, "freed_bytes": freed}

        raise ValueError(f"Unsupported recycle bin action: {action!r}")

    def cleanup_unused_takes(self) -> dict[str, Any]:
        """Backward-compatible wrapper: recycle all unprotected historical takes and clean project.json."""
        res = self.cleanup_items(scope="project_unused", mode="recycle")
        return {
            "recycled_count": len(res["deleted_takes"]),
            "recycled_files": list(res.get("recycled_files") or ()),
            "deleted_takes": list(res["deleted_takes"]),
            "freed_bytes": res["freed_bytes"],
            "batch_id": res["batch_id"],
        }


def compute_continuation_depth(
    shots: list[dict[str, Any]],
    curr_idx: int,
    is_cont: bool,
    has_seed: bool = False,
) -> int:
    """Count uninterrupted `@continue@` streak depth ending at `curr_idx` (1 = fresh cut / Shot 1)."""
    if not is_cont:
        return 1
    depth = 1
    idx = int(curr_idx)
    while idx > 1 and (idx - 1) < len(shots) and bool(shots[idx - 1].get("is_continuation", False)):
        depth += 1
        idx -= 1
    if idx == 1 and has_seed and shots and bool(shots[0].get("is_continuation", False)):
        depth += 1
    return depth


def load_scene_anchor_clean_ref(
    store: Optional[VaultStore],
    shots: list[dict[str, Any]],
    curr_idx: int,
    is_cont: bool,
    parent_tid: str,
    prev_clean_images: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Walk backward to the opening shot of the current continuous scene and return its 4-frame clean tail."""
    fallback = (
        prev_clean_images[-4:].float().contiguous()
        if isinstance(prev_clean_images, torch.Tensor) and int(prev_clean_images.shape[0]) > 0 and float(prev_clean_images.abs().max().item()) > 0.0
        else None
    )
    if not is_cont or store is None:
        return fallback

    anchor_idx = max(0, int(curr_idx) - 1)
    while anchor_idx > 1 and (anchor_idx - 1) < len(shots) and bool(shots[anchor_idx - 1].get("is_continuation", False)):
        anchor_idx -= 1
    if anchor_idx == 1 and shots and bool(shots[0].get("is_continuation", False)) and store.get_active_take_id(0):
        anchor_idx = 0

    anc_tid = store.get_active_take_id(anchor_idx)
    if not anc_tid or anc_tid == parent_tid:
        return fallback
    anc_ctx = store.load_take_context(anc_tid)
    if anc_ctx is not None and "clean_tail" in anc_ctx["tensors"]:
        return anc_ctx["tensors"]["clean_tail"][-4:].float().contiguous()
    return fallback


def _seam_repair_sig(sr: Optional[dict[str, Any]]) -> str:
    if not isinstance(sr, dict):
        return "original"
    on = bool(sr.get("enabled", False)) and str(sr.get("choice") or "repaired") == "repaired" and str(sr.get("mode") or "rife") != "cut"
    if not on:
        return "original"
    return f"{sr.get('mode','rife')}:{sr.get('rife_bridge_frames',5)}:{bool(sr.get('preserve_duration',False))}:{sr.get('take_id','')}"


def compute_master_sync_status(
    store: VaultStore,
    shots_plan: list[dict[str, Any]],
    project: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Compare current project active takes & seam repairs against latest exported master MP4/manifest."""
    proj = project if isinstance(project, dict) else store.load_project()
    proj_name = str(proj.get("project_name") or os.path.basename(store.project_root))
    primary_mp4 = os.path.join(store.master_dir, f"{proj_name}_master.mp4")
    target_mp4 = primary_mp4 if os.path.isfile(primary_mp4) else ""
    if not target_mp4 and os.path.isdir(store.master_dir):
        cands = [
            os.path.join(store.master_dir, fn)
            for fn in os.listdir(store.master_dir)
            if fn.lower().endswith((".mp4", ".mov")) and ".tmp." not in fn.lower()
        ]
        cands = [p for p in cands if os.path.isfile(p)]
        if cands:
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            target_mp4 = cands[0]

    shots_state = proj.get("shots") or {}
    planned_indices = [int(s["shot_index"]) for s in shots_plan] if shots_plan else sorted(int(k) for k in shots_state.keys() if str(k).isdigit())
    rendered_indices = [i for i in planned_indices if (shots_state.get(str(i)) or {}).get("active_take_id")]

    if not target_mp4:
        return {
            "has_master": False,
            "master_rel": "",
            "master_name": "",
            "master_mtime": 0.0,
            "is_outdated": bool(rendered_indices),
            "outdated_shots": rendered_indices[:8],
            "outdated_reasons": ["尚未导出总成片"] if rendered_indices else [],
            "outdated_reasons_en": ["Master video not exported yet"] if rendered_indices else [],
        }

    master_mtime = float(os.path.getmtime(target_mp4))
    master_name = os.path.basename(target_mp4)
    master_rel = f"master/{master_name}"
    base_no_ext = os.path.splitext(target_mp4)[0]
    manifest_path = f"{base_no_ext}.manifest.json"
    manifest: Optional[dict[str, Any]] = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception:
            manifest = None

    outdated_shots: list[int] = []
    outdated_reasons: list[str] = []
    outdated_reasons_en: list[str] = []
    m_shots = (manifest or {}).get("shots") if isinstance(manifest, dict) else None

    for idx in rendered_indices:
        rec = shots_state.get(str(idx)) or {}
        act_tid = str(rec.get("active_take_id") or "")
        cur_sr_sig = _seam_repair_sig(rec.get("seam_repair"))
        if isinstance(m_shots, dict):
            m_rec = m_shots.get(str(idx))
            if not isinstance(m_rec, dict):
                outdated_shots.append(idx)
                outdated_reasons.append(f"#{idx} ({act_tid} 未合入)")
                outdated_reasons_en.append(f"#{idx} ({act_tid} not merged)")
            elif str(m_rec.get("take_id") or "") != act_tid:
                outdated_shots.append(idx)
                outdated_reasons.append(f"#{idx} (已换为 {act_tid})")
                outdated_reasons_en.append(f"#{idx} (switched to {act_tid})")
            elif str(m_rec.get("seam_repair_sig") or "original") != cur_sr_sig:
                outdated_shots.append(idx)
                outdated_reasons.append(f"#{idx} (修复配置已变更)")
                outdated_reasons_en.append(f"#{idx} (seam repair changed)")
        else:
            # Legacy project without sidecar manifest: compare active take file mtime & active seam repair toggle
            tmeta = (rec.get("takes") or {}).get(act_tid) or {}
            vrel = str(tmeta.get("video_rel_path") or f"shots/{act_tid}.mp4")
            try:
                vabs = ensure_inside_root(store.project_root, vrel)
            except ValueError:
                vabs = ""
            if vabs and os.path.isfile(vabs) and os.path.getmtime(vabs) > master_mtime + 1.0:
                outdated_shots.append(idx)
                outdated_reasons.append(f"#{idx} ({act_tid} 新于旧成片)")
                outdated_reasons_en.append(f"#{idx} ({act_tid} newer than master)")
            elif cur_sr_sig != "original":
                outdated_shots.append(idx)
                outdated_reasons.append(f"#{idx} (已开启修复待合入)")
                outdated_reasons_en.append(f"#{idx} (seam repair pending)")

    return {
        "has_master": True,
        "master_rel": master_rel,
        "master_name": master_name,
        "master_mtime": master_mtime,
        "is_outdated": bool(outdated_shots),
        "outdated_shots": outdated_shots,
        "outdated_reasons": outdated_reasons,
        "outdated_reasons_en": outdated_reasons_en,
    }


def load_next_shot_head_ref(
    store: Optional[VaultStore],
    next_shot_index: int,
    head_frames: int,
    fps: int = FPS_DEFAULT,
) -> Optional[dict[str, Any]]:
    """Decode ONLY the first `head_frames` (1, 5, or 22) clean delivered frames (+ optional 32kHz head audio)
    from `next_shot_index`'s active delivered MP4 (`shots/shot_xxxx_take_yyy.mp4`).

    Delivered MP4s are ALREADY post-`ShotVaultSeamTrim`, so their head contains 100% clean generated frames
    with ZERO `#106` chroma noise.
    """
    req_f = int(head_frames or 0)
    if store is None or req_f not in (1, 5, 22):
        return None
    proj = store.load_project()
    rec = (proj.get("shots") or {}).get(str(int(next_shot_index))) or {}
    next_tid = str(rec.get("active_take_id") or "").strip()
    if not next_tid:
        return None
    tmeta = (rec.get("takes") or {}).get(next_tid) or {}
    vrel = str(tmeta.get("video_rel_path") or f"shots/{next_tid}.mp4")
    try:
        vabs = ensure_inside_root(store.project_root, vrel)
    except ValueError:
        return None
    if not os.path.isfile(vabs):
        return None

    try:
        import av  # type: ignore
        import numpy as np

        frames_list: list[np.ndarray] = []
        with av.open(vabs) as container:
            v_stream = next((s for s in container.streams if s.type == "video"), None)
            if v_stream is None:
                return None
            for fr in container.decode(video=0):
                frames_list.append(fr.to_ndarray(format="rgb24"))
                if len(frames_list) >= req_f:
                    break
        if not frames_list:
            return None
        actual_f = len(frames_list)
        if actual_f < req_f:
            # Safe fallback to largest valid H3 tail count available
            if actual_f >= 5:
                req_f = 5
            else:
                req_f = 1
            frames_list = frames_list[:req_f]
        arr = np.stack(frames_list, axis=0)
        head_images = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0

        head_audio_32k = None
        if req_f == 22:
            # Extract first 20 frames (0.83s, keeping 2f guard band on the left of tail anchor) at 32kHz
            need_sec = 20.0 / float(max(1, int(fps)))
            aud_wav = None
            ctx = store.load_take_context(next_tid)
            if ctx is not None and "delivery_audio_48k" in ctx["tensors"]:
                aud_wav = resample_audio(ctx["tensors"]["delivery_audio_48k"].float(), DELIVERY_AUDIO_SR, AUDIO_VAE_SR)
            else:
                chunks = []
                with av.open(vabs) as container:
                    a_stream = next((s for s in container.streams if s.type == "audio"), None)
                    if a_stream is not None:
                        src_sr = int(a_stream.sample_rate or DELIVERY_AUDIO_SR)
                        max_src = int(round((need_sec + 0.2) * src_sr))
                        got = 0
                        for afr in container.decode(audio=0):
                            a_np = afr.to_ndarray()
                            nb_ch = int(getattr(afr.layout, "nb_channels", a_np.shape[0] if a_np.ndim > 1 else 1) or 1)
                            if not afr.format.is_planar and nb_ch > 1:
                                a_np = a_np.reshape(-1, nb_ch).T
                            elif a_np.ndim == 1:
                                a_np = a_np[None, :]
                            t_a = torch.from_numpy(np.ascontiguousarray(a_np)).float()
                            if t_a.abs().max() > 1.5:
                                t_a = t_a / 32768.0
                            if t_a.shape[0] == 1:
                                t_a = t_a.repeat(2, 1)
                            elif t_a.shape[0] > 2:
                                t_a = t_a[:2]
                            chunks.append(t_a)
                            got += int(t_a.shape[1])
                            if got >= max_src:
                                break
                        if chunks:
                            aud_wav = resample_audio(torch.cat(chunks, dim=1), src_sr, AUDIO_VAE_SR)
            if aud_wav is not None and aud_wav.shape[1] > 0:
                need_samps = max(800, round(need_sec * AUDIO_VAE_SR))
                sl = aud_wav[:, :need_samps]
                if sl.shape[1] < need_samps:
                    sl = F.pad(sl, (0, need_samps - sl.shape[1]))
                head_audio_32k = {"waveform": sl.unsqueeze(0).contiguous(), "sample_rate": AUDIO_VAE_SR}

        return {
            "next_shot_index": int(next_shot_index),
            "next_take_id": next_tid,
            "head_frames": int(req_f),
            "images": head_images.contiguous(),
            "audio_32k": head_audio_32k,
        }
    except Exception:
        return None
