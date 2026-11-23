# ShotVault Save · 分镜入库与自动重合

**English** | Atomically archives each rendered shot take (`shots/shot_xxxx_take_yyy.mp4`), thumbnail, 3s seam preview, and `.h3ctx.safetensors` continuation context.
- Records `bridged_next_take_id` after a bidirectional bridge reroll so the downstream shot automatically stays green `VALID`.
- When `auto_assemble_after_reroll` is enabled during a single-shot reroll, immediately refreshes `master.mp4`, `.manifest.json`, and `FCPXML`.

---

**简体中文** | 将每镜成片（`shots/shot_xxxx_take_yyy.mp4`）、缩略图、3 秒接缝预览与 `.h3ctx.safetensors` 上下文原子落盘归档。
- 双向桥接重抽完成后自动记录 `bridged_next_take_id`，使下游已渲染镜头直接保持绿标 `VALID`。
- 单段定向重抽开启 `auto_assemble_after_reroll` 时，入库后自动刷新 `master.mp4`、`.manifest.json` 校验清单与 `FCPXML`。
