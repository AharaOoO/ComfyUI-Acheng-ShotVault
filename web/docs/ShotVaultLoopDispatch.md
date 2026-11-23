# ShotVault Loop Dispatch · 循环调度与双向取帧

**English** | Stateless per-iteration dispatcher for external loop graphs (e.g. `easy forLoopStart/End`).
- Automatically skips already-`VALID` shots when resuming an interrupted run.
- Honors `ShotVaultPlan`'s **Single-Shot Reroll Lock** (`honor_director_retry_lock = True`), running only the targeted shot (`total_shots = 1`).
- When **Bidirectional Tail Bridge (`1 / 5 / 22f`)** is enabled, decodes clean post-trim head frames directly from the next shot's delivered MP4 (zero chroma-noise contamination) and aligns exact H3 `17k+5` grid target frames.

---

**简体中文** | 面向外部循环架构（如 `easy forLoopStart/End`）的无状态逐镜分发节点。
- 断点续跑时自动跳过已 `VALID` 镜头，避免重复消耗算力。
- 开启 `honor_director_retry_lock` 后自动响应分镜规划台的单段重抽锁定，仅运行目标镜头（`total_shots = 1`）。
- 启用 **双头桥接 (`1 / 5 / 22f`)** 时，直接从下一段已裁头交付成品 MP4 解码纯净首部帧（零彩噪污染），并严格对齐 H3 `17k+5` 网格目标帧数。
