# ShotVault Assemble · 单遍 PCM 总装

**English** | Zero-drift single-pass PCM/AAC master video assembler with optional DaVinci Resolve `FCPXML` export.
- Atomically writes a sidecar `.manifest.json` alongside every `master/*.mp4` recording exact active takes and seam-repair signatures for frontend staleness checks.
- **`stale_policy`**:
  - `defer_until_complete`: waits automatically while unrendered/STALE shots remain in the loop.
  - `allow_with_warning`: assembles all active takes immediately.
  - `block_and_report`: strict validation gate.

---

**简体中文** | 零音画漂移单遍 PCM/AAC 总成片封装节点，支持同步导出达芬奇 `FCPXML` 工程。
- 每次导出 `master/*.mp4` 均原子写入伴生 `.manifest.json`，记录全部采用的 Take 与接缝修复签名，供前端精准检测过期状态。
- **`stale_policy`**：
  - `defer_until_complete`：循环未跑完或存在标红镜头时自动延迟封装；
  - `allow_with_warning`：按当前所选 Take 立即合并导出；
  - `block_and_report`：严格校验阻断。
