# Third-Party Notices & Attributions

`ComfyUI-Acheng-ShotVault` respects open-source licenses and intellectual property rights. This repository contains original author-owned code plus the MIT-licensed lightweight `rife422lite_fp32.onnx` model weight for out-of-the-box local seam interpolation, and does **not** vendor third-party ComfyUI node repositories or proprietary binary assets.

The example workflow and optional runtime features interoperate with the following independent open-source projects, whose copyrights and licenses belong exclusively to their respective authors:

| Project / Component | Author / Maintainer | License | Relationship to ShotVault |
| :--- | :--- | :--- | :--- |
| **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** | comfyanonymous / Comfy Org | GPL-3.0 | Host execution runtime |
| **[ComfyUI-H3-Context-Noise](https://github.com/beijinren/ComfyUI-H3-Context-Noise)** | beijinren (Copyright © 2026 beijinren) | MIT | External companion node used at `#106` (`MiniMaxH3ContextTaperNoise`) in the example workflow; **not bundled** in this repo — install separately from upstream |
| **[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)** | kijai | GPL-3.0 | External companion nodes (`GetNode`, `SetNode`, `MiniMaxLowVRAMAttention`, `ModelPreviewOverrideKJ`, etc.) |
| **[ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)** | yolain | GPL-3.0 | External loop & flow-control companion nodes (`easy forLoopStart/End`, `easy ifElse`, etc.) |
| **[ComfyUI-Custom-Scripts](https://github.com/pysssss/ComfyUI-Custom-Scripts)** | pysssss | MIT | External UI companion node (`ShowText|pysssss`) |
| **[Goohaitools-comfyui](https://github.com/goohai/Goohaitools-comfyui)** | goohai | See upstream | External slider control node (`GoohaiUniversalSlider`) used in example workflow |
| **comfyui-minimax-h3-audio-T8** | Community contributors | See upstream | External MiniMax H3 LoRA compatibility & semantic bridge nodes used in example workflow |
| **[ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)** & **[Practical-RIFE](https://github.com/hzwer/ECCV2022-RIFE)** | Fannovel16 / hzwer (Megvii) | MIT | Optional local `.pth` RIFE backend; bundled `seam_bridge/models/rife422lite_fp32.onnx` is derived from MIT-licensed Practical-RIFE v4.22 lite |
| **PyTorch / NumPy / OpenCV / PyAV / safetensors / Pillow / send2trash / ONNX Runtime** | Respective upstream teams | BSD / Apache-2.0 / MIT | Standard Python libraries declared in `requirements.txt` |

### Clean-Room Implementation Statement (`seam_bridge/`)

The `seam_bridge/` module (`AchengSeamBridge`, `AchengFuzzPuppyClipMerge`, and `fuzzpuppy_engine.py`) is an independent local Python/OpenCV/ONNX implementation written from scratch for local ComfyUI pipelines. It contains **zero** copied third-party node code, **zero** web service assets, and **zero** telemetry/network uploads; `seam_bridge/models/rife422lite_fp32.onnx` is bundled solely for local offline frame interpolation under the Practical-RIFE MIT License. Any technical terminology referencing public clip-merge concepts is strictly descriptive of local optical-flow / color-alignment / frame-bridging algorithms and implies no affiliation with or endorsement by any external service.
