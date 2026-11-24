# RIFE ONNX Model Directory

This directory includes the bundled `rife422lite_fp32.onnx` (~18.5 MB) lightweight frame-interpolation weight (based on [Practical-RIFE](https://github.com/hzwer/ECCV2022-RIFE), MIT License) so `AchengSeamBridge` and `AchengFuzzPuppyClipMerge` work out of the box without requiring external node installations.

### Lookup Order

At runtime, the engine checks for RIFE backends in the following order:

1. `ComfyUI/custom_nodes/ComfyUI-Acheng-ShotVault/seam_bridge/models/rife422lite_fp32.onnx` (bundled)
2. `ComfyUI/models/rife/rife422lite_fp32.onnx` (optional shared override)
3. Locally installed `ComfyUI-Frame-Interpolation` RIFE VFI `.pth` checkpoints (`rife426.pth`, `rife49.pth`, `rife47.pth`)

> **Runtime Note:** Ensure `onnxruntime` (or `onnxruntime-gpu` for CUDA acceleration) is installed in your ComfyUI Python environment to execute the ONNX backend.
