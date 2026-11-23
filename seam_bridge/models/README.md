# Optional RIFE ONNX Model Directory

This repository does **not** redistribute third-party binary model weights.

`AchengSeamBridge` and `AchengFuzzPuppyClipMerge` work out of the box using:
1. Locally installed RIFE weights from `ComfyUI-Frame-Interpolation` (if installed), or
2. Built-in OpenCV DIS optical-flow / local geometric alignment fallback (zero external weights required).

### Optional: Using an ONNX RIFE Model

If you obtain a compatible `rife422lite_fp32.onnx` weight locally under its applicable license, place it in **either** of the following paths:

- `ComfyUI/custom_nodes/ComfyUI-Acheng-ShotVault/seam_bridge/models/rife422lite_fp32.onnx`
- `ComfyUI/models/rife/rife422lite_fp32.onnx`

The engine will automatically detect and load it via `onnxruntime` (CUDA or CPU provider) on next run.
