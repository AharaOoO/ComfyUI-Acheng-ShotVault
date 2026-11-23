# ShotVault Seam Trim · 精确裁头与 CIE Lab 锁色

**English** | Trims the continuation overlap head (`22 / 39 / 56f`), locks color continuity via CIE Lab MKL linear matching, and runs local seam QC diagnostics.
- Supports `Ctrl + B` bypass safely (0th port `IMAGE -> IMAGE`).
- Set `trim_head_frames = -1` and `keep_frames = -1` to follow `shot_pipe` automatically (including bidirectional tail-bridge trim math).

---

**简体中文** | 负责精准裁切续接引导重叠区（`22 / 39 / 56f`）、执行 CIE Lab MKL 线性锁色与接缝跳变质检。
- 第 0 端口保持 `IMAGE -> IMAGE` 同构对齐，支持 `Ctrl + B` 原生旁路。
- `trim_head_frames = -1` 且 `keep_frames = -1` 时自动跟随 `shot_pipe` 精确计算（含双向桥接裁切）。
