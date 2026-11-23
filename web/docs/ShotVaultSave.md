# ShotVault Save (双轨隔离归档)

`ShotVaultSave` 将成片与接续上下文物理隔离落盘，并在队列头安全推进下一镜。

## 关键端口与防踩坑指南
- **`final_images` vs `clean_images` 物理隔离**：
  - 将超分 / LUT / 锐化之后的画面接 `final_images`（写入成片 `shots/*.mp4`）。
  - 将后处理之前的 `[N4] ShotVaultSeamTrim.images` 单独接入 `clean_images`（写入 `_vault/*.h3ctx.safetensors`）。如此哪怕 2x 超分也不会污染下一镜底噪！
- **`sampled_latent`**: 接入二采采样器输出，自动无损抽出末 `12` 步裸 AV 尾潜变量。
- **`auto_queue_next`**: 开启时按剩余待跑镜从服务端 **`PromptServer` 队列头** 自动续排；点击 ComfyUI `Cancel` 或报错瞬间熔断。
