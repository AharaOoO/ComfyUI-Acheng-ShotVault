"""Zero-frame-delta FP16 RIFE (`rife49.pth`) micro-bridge with DIS optical-flow fallback."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

_RIFE_CACHE: dict[tuple[str, str], tuple[Any, torch.device]] = {}
_ARCH_MOD_CACHE: Any = None
_PREFERRED_CKPTS = ("rife49.pth", "rife426.pth", "rife47.pth")


def _comfy_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_rife_arch_path() -> Optional[str]:
    root = _comfy_root()
    candidates = (
        root / "custom_nodes" / "ComfyUI-Frame-Interpolation" / "vfi_models" / "rife" / "rife_arch.py",
        Path(__file__).resolve().parent / "rife_arch.py",
    )
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _get_rife_arch_mod():
    global _ARCH_MOD_CACHE
    if _ARCH_MOD_CACHE is not None:
        return _ARCH_MOD_CACHE
    arch_path = _find_rife_arch_path()
    if not arch_path:
        return None
    try:
        import sys
        import types

        str_root = str(_comfy_root())
        if str_root not in sys.path:
            sys.path.insert(0, str_root)
        try:
            import comfy.model_management  # noqa: F401
        except ImportError:
            fake_comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
            fake_mm = types.ModuleType("comfy.model_management")
            fake_mm.get_torch_device = lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            fake_comfy.model_management = fake_mm
            sys.modules["comfy.model_management"] = fake_mm

        spec = importlib.util.spec_from_file_location("shotvault_rife_arch", arch_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["shotvault_rife_arch"] = mod
        spec.loader.exec_module(mod)
        _ARCH_MOD_CACHE = mod
        return mod
    except Exception:
        return None


def _find_best_ckpt(ckpt_name: str = "") -> Optional[tuple[str, str]]:
    root = _comfy_root()
    search_dirs = (
        root / "custom_nodes" / "ComfyUI-Frame-Interpolation" / "ckpts" / "rife",
        root / "models" / "rife",
        root / "models" / "vfi",
        Path(__file__).resolve().parent / "ckpts",
    )
    names = ([ckpt_name] if ckpt_name else []) + list(_PREFERRED_CKPTS)
    for name in names:
        for d in search_dirs:
            p = d / name
            if p.is_file():
                return str(p), name
    for d in search_dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.pth")):
                return str(f), f.name
    return None


def _infer_arch_ver(ckpt_name: str) -> str:
    low = ckpt_name.lower()
    if "426" in low:
        return "4.26"
    if "417" in low:
        return "4.17"
    if "sudo" in low or "40" in low:
        return "4.0"
    return "4.7"


def load_cached_rife(ckpt_name: str = ""):
    """Load singleton IFNet onto `rife_mod.device` (FP16 on CUDA, FP32 on CPU)."""
    mod = _get_rife_arch_mod()
    found = _find_best_ckpt(ckpt_name)
    if mod is None or found is None:
        return None
    ckpt_path, resolved_name = found
    target_device = getattr(mod, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype_key = "float16" if target_device.type == "cuda" else "float32"
    cache_key = (ckpt_path, dtype_key)
    if cache_key in _RIFE_CACHE:
        return _RIFE_CACHE[cache_key]

    try:
        model = mod.IFNet(arch_ver=_infer_arch_ver(resolved_name)).to(target_device)
        sd = torch.load(ckpt_path, map_location=target_device)
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
        model.eval()
        if target_device.type == "cuda":
            model.half()
        _RIFE_CACHE[cache_key] = (model, target_device)
        return _RIFE_CACHE[cache_key]
    except Exception:
        return None


def _dis_optical_flow_bridge(frame_a: torch.Tensor, frame_b: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Bidirectional OpenCV DIS optical-flow fallback returning `[num_frames, H, W, 3]`."""
    import cv2

    a_np = frame_a.squeeze(0)[..., :3].detach().cpu().float().clamp(0.0, 1.0).numpy()
    b_np = frame_b.squeeze(0)[..., :3].detach().cpu().float().clamp(0.0, 1.0).numpy()
    h, w = a_np.shape[:2]
    gray_a = cv2.cvtColor((a_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor((b_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f_ab = dis.calc(gray_a, gray_b, None)
    f_ba = dis.calc(gray_b, gray_a, None)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    frames = []
    for k in range(num_frames):
        t_k = (k + 1.0) / float(num_frames + 1)
        wa = cv2.remap(a_np, gx + t_k * f_ab[..., 0], gy + t_k * f_ab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(b_np, gx + (1.0 - t_k) * f_ba[..., 0], gy + (1.0 - t_k) * f_ba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frames.append(np.clip((1.0 - t_k) * wa + t_k * wb, 0.0, 1.0))
    return torch.from_numpy(np.stack(frames, axis=0)).to(device=frame_a.device, dtype=frame_a.dtype)


def synthesize_bridge_frames(
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
    num_frames: int,
    ckpt_name: str = "",
) -> tuple[torch.Tensor, str]:
    """Synthesize `num_frames` sharp optical-flow frames between `frame_a` and `frame_b` (`0..1`)."""
    out_device, out_dtype = frame_a.device, frame_a.dtype
    fa = frame_a.unsqueeze(0) if frame_a.ndim == 3 else frame_a[:1]
    fb = frame_b.unsqueeze(0) if frame_b.ndim == 3 else frame_b[:1]
    _, h, w, _ = fa.shape
    if num_frames <= 0:
        return torch.zeros((0, h, w, 3), device=out_device, dtype=out_dtype), "none"
    if fb.shape[1] != h or fb.shape[2] != w:
        fb = F.interpolate(fb[..., :3].permute(0, 3, 1, 2).float(), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

    loaded = load_cached_rife(ckpt_name=ckpt_name)
    if loaded is not None:
        model, target_device = loaded
        try:
            pad_h = (128 - (h % 128)) % 128
            pad_w = (128 - (w % 128)) % 128
            t_dtype = torch.float16 if target_device.type == "cuda" else torch.float32
            img0 = F.pad(fa[..., :3].permute(0, 3, 1, 2).to(device=target_device, dtype=t_dtype).clamp(0.0, 1.0), (0, pad_w, 0, pad_h), mode="replicate")
            img1 = F.pad(fb[..., :3].permute(0, 3, 1, 2).to(device=target_device, dtype=t_dtype).clamp(0.0, 1.0), (0, pad_w, 0, pad_h), mode="replicate")

            synth = []
            with torch.inference_mode():
                for k in range(int(num_frames)):
                    t_k = (k + 1.0) / float(num_frames + 1)
                    out = model(
                        img0, img1, timestep=float(t_k),
                        scale_list=[8.0, 4.0, 2.0, 1.0],
                        training=False, fastmode=True, ensemble=False,
                    )
                    synth.append(out[:, :, :h, :w].permute(0, 2, 3, 1).clamp(0.0, 1.0))
            stacked = torch.cat(synth, dim=0).to(device=out_device, dtype=out_dtype)
            return stacked, "rife"
        except Exception:
            pass

    return _dis_optical_flow_bridge(fa, fb, int(num_frames)), "dis_flow_fallback"


def apply_zero_delta_rife_micro_bridge(
    images: torch.Tensor,
    ref_tail: Optional[torch.Tensor],
    mode: str = "auto_jump_rife",
    max_bridge_frames: int = 4,
    jump_spike_threshold: float = 1.65,
    gate_info: Optional[dict[str, Any]] = None,
    known_continuation: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Zero-frame-delta seam bridge: replaces `images[:k_bridge]` with RIFE transitions between
    `ref_tail[-1]` and `images[k_bridge]` ONLY when needed (or `mode == "always_rife"`).
    Preserves exact `[F, H, W, C]` shape and audio sync (0 frame delta).
    """
    info: dict[str, Any] = {
        "applied_rife": False,
        "rife_frames": 0,
        "backend": "none",
        "spike_ratio": 0.0,
        "d_seam": 0.0,
        "d_intra": 0.0,
        "has_head_freeze": False,
        "reason": "skip",
    }
    mode_s = str(mode or "auto_jump_rife").strip()
    if mode_s == "off" or ref_tail is None or not isinstance(ref_tail, torch.Tensor):
        info["reason"] = "disabled_or_no_ref"
        return images, info
    if images.ndim != 4 or int(images.shape[0]) < 3 or ref_tail.ndim != 4 or int(ref_tail.shape[0]) == 0 or int(max_bridge_frames) <= 0:
        info["reason"] = "too_short"
        return images, info
    if isinstance(gate_info, dict):
        if gate_info.get("is_flash"):
            info["reason"] = "impact_flash_protected"
            return images, info
        if gate_info.get("same_scene_safe") is False:
            info["reason"] = "extreme_cut_protected" if known_continuation else "hard_cut_protected"
            return images, info

    n_frames, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    h_s = max(24, round(h * 128.0 / max(1, w)))
    curr_probe = F.interpolate(images[:min(7, n_frames), ..., :3].float().movedim(-1, 1), size=(h_s, 128), mode="area")
    ref_probe = F.interpolate(ref_tail[-min(4, int(ref_tail.shape[0])):, ..., :3].to(device=images.device).float().movedim(-1, 1), size=(h_s, 128), mode="area")

    d_seam = float((curr_probe[0] - ref_probe[-1]).abs().mean().item())
    intra_diffs: list[torch.Tensor] = []
    if curr_probe.shape[0] > 1:
        intra_diffs.append((curr_probe[1:] - curr_probe[:-1]).abs().mean(dim=(1, 2, 3)))
    if ref_probe.shape[0] > 1:
        intra_diffs.append((ref_probe[1:] - ref_probe[:-1]).abs().mean(dim=(1, 2, 3)))
    d_intra = float(torch.cat(intra_diffs, dim=0).median().item()) if intra_diffs else 0.01
    curr_01_diff = float((curr_probe[1] - curr_probe[0]).abs().mean().item()) if curr_probe.shape[0] > 1 else d_intra
    has_head_freeze = (d_seam > 0.008) and (curr_01_diff < 0.35 * max(d_intra, 0.008))
    spike_ratio = d_seam / max(d_intra, 0.004)
    info.update({
        "spike_ratio": round(spike_ratio, 3),
        "d_seam": round(d_seam, 5),
        "d_intra": round(d_intra, 5),
        "has_head_freeze": bool(has_head_freeze),
    })

    if mode_s != "always_rife" and d_seam > 0.16:
        info["reason"] = "camera_cut_too_wide_for_rife"
        return images, info

    trigger = (mode_s == "always_rife") or (spike_ratio >= float(jump_spike_threshold) and d_seam >= 0.012) or has_head_freeze
    if not trigger:
        info["reason"] = "smooth_passthrough"
        return images, info

    max_k = min(int(max_bridge_frames), n_frames - 1)
    if mode_s == "always_rife":
        k_bridge = max_k
    elif d_intra > 0.045:
        k_bridge = min(max_k, 2)
    else:
        k_bridge = min(max_k, max(2, round(min(float(max_bridge_frames), 1.5 + 0.9 * spike_ratio))))
    if k_bridge < 1:
        info["reason"] = "zero_k_bridge"
        return images, info

    ref_last = ref_tail[-1:, ..., :3].to(device=images.device, dtype=images.dtype)
    if ref_last.shape[1] != h or ref_last.shape[2] != w:
        ref_last = F.interpolate(ref_last.float().movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1).to(dtype=images.dtype)

    synth, backend = synthesize_bridge_frames(ref_last[0], images[k_bridge, ..., :3], k_bridge)
    out = images.clone()
    out[:k_bridge, ..., :3] = synth.to(device=images.device, dtype=images.dtype)
    info.update({
        "applied_rife": True,
        "rife_frames": int(k_bridge),
        "backend": backend,
        "reason": "always_rife" if mode_s == "always_rife" else ("freeze_heal" if has_head_freeze else "jump_heal"),
    })
    return out, info
