# ShotVault Assemble (单遍PCM总装)

`ShotVaultAssemble` 可直接串联于主工作流末尾，也可独立拖出单独执行一键总装导出。

## 关键参数
- **`stale_policy`**:
  - `defer_until_complete`（默认）：只要全片还有未跑完或 `STALE` 镜头、或队列仍在推进，**自动静默延后跳过而不报错**；当最后一镜入库后自动触发整片 `-c:v copy` 无损拼合与单遍 `f32le` PCM 封箱。
  - `block_and_report`：严格拦截未完成或标红镜头并抛出报错详情。
  - `allow_with_warning`：带警告强制合成当前已出片镜头。
- **`audio_crossfade_ms` / `post_mux_offset_ms`**: 硬切微渐变防咔哒与免重跑音画微调。
- **`export_fcpxml`**: 一键导出达芬奇 Resolve FCPXML v1.9 时间线工程。
