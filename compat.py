"""Environment compatibility helpers for ComfyUI-Universal-ShotVault."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import shutil
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger("ShotVault")

_INVALID_WIN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_RESERVE_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


def sanitize_project_name(name: str, fallback: str = "MyH3Film") -> str:
    """Sanitize a project folder name to block path traversal and Windows reserved names."""
    text = str(name or "").replace("..", "_").strip(" .")
    text = _INVALID_WIN_CHARS.sub("_", text).strip(" ._")
    if not text or text.upper() in _RESERVE_NAMES:
        return fallback
    return text[:96]


def resolve_output_root(project_name: str, override_base: Optional[str] = None) -> str:
    """Resolve and sanitize `<output_dir>/ShotVault/<project_name>`."""
    safe_name = sanitize_project_name(project_name)
    if override_base:
        base_dir = os.path.abspath(override_base)
    else:
        base_dir = None
        try:
            import folder_paths  # type: ignore

            base_dir = os.path.abspath(folder_paths.get_output_directory())
        except Exception:
            base_dir = os.path.abspath(os.path.join(os.getcwd(), "output"))
    root = os.path.abspath(os.path.join(base_dir, "ShotVault", safe_name))
    expected_prefix = os.path.abspath(os.path.join(base_dir, "ShotVault")) + os.sep
    if not root.startswith(expected_prefix):
        raise ValueError(f"ShotVault: unsafe project path rejected: {root!r}")
    return root


def ensure_inside_root(root_dir: str, rel_or_abs_path: str) -> str:
    """Resolve a path strictly inside `root_dir`, raising ValueError on escape."""
    root_abs = os.path.abspath(root_dir)
    if os.path.isabs(rel_or_abs_path):
        candidate = os.path.abspath(rel_or_abs_path)
    else:
        candidate = os.path.abspath(os.path.join(root_abs, rel_or_abs_path))
    if candidate != root_abs and not candidate.startswith(root_abs + os.sep):
        raise ValueError(f"ShotVault: path escapes project root: {rel_or_abs_path!r}")
    return candidate


def resolve_ffmpeg_exe() -> str:
    """Find a working FFmpeg executable, preferring imageio_ffmpeg portable binary."""
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    which_exe = shutil.which("ffmpeg")
    if which_exe and os.path.isfile(which_exe):
        return which_exe
    raise RuntimeError("ShotVault: FFmpeg executable not found via imageio_ffmpeg or PATH.")


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample floating-point waveform `[..., samples]` from `orig_sr` to `target_sr`."""
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if not waveform.is_floating_point():
        waveform = waveform.float()
    if orig_sr == target_sr or waveform.shape[-1] == 0:
        return waveform
    try:
        import comfy.audio  # type: ignore

        return comfy.audio.resample(waveform, orig_sr, target_sr)
    except Exception:
        pass
    try:
        import torchaudio  # type: ignore

        return torchaudio.functional.resample(waveform, orig_sr, target_sr)
    except Exception:
        pass
    orig_shape = waveform.shape
    flat = waveform.reshape(-1, 1, orig_shape[-1]).float()
    new_len = max(1, round(orig_shape[-1] * target_sr / float(orig_sr)))
    out = F.interpolate(flat, size=new_len, mode="linear", align_corners=False)
    return out.reshape(*orig_shape[:-1], new_len).to(dtype=waveform.dtype)


def safe_recycle(path: str, recycle_dir: str) -> Optional[str]:
    """Send file to OS recycle bin via send2trash, falling back to `_recycle_bin`."""
    if not path or not os.path.exists(path):
        return None
    abs_path = os.path.abspath(path)
    try:
        import send2trash  # type: ignore

        send2trash.send2trash(abs_path)
        return "recycle_bin"
    except Exception as exc:
        LOGGER.debug("ShotVault send2trash fallback (%s): %s", abs_path, exc)
    os.makedirs(recycle_dir, exist_ok=True)
    stamp = int(time.time() * 1000)
    dst = os.path.join(recycle_dir, f"{stamp}_{os.path.basename(abs_path)}")
    shutil.move(abs_path, dst)
    return dst


def sha256_text(text: str) -> str:
    """Return hex SHA-256 of normalized UTF-8 string."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def ensure_audio_feather_patch() -> bool:
    """Apply installed ComfyUI-MiniMax-H3-Audio-Guide-Feather layout patch if present."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.dirname(pkg_dir)
    compat_py = os.path.join(custom_nodes_dir, "ComfyUI-MiniMax-H3-Audio-Guide-Feather", "compat.py")
    if not os.path.isfile(compat_py):
        return False
    mod_name = "_shotvault_feather_compat"
    try:
        mod = sys.modules.get(mod_name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(mod_name, compat_py)
            if spec is None or spec.loader is None:
                return False
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        fn = getattr(mod, "ensure_h3_audio_feather_compat", None)
        if callable(fn):
            fn()
            return True
    except Exception as exc:
        LOGGER.debug("ShotVault audio feather compat check skipped: %s", exc)
    return False
