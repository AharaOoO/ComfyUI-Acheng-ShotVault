# ShotVault Window Guider · 零劣化窗口引导

**English** | Context-window & bidirectional continuation guider for 1st-pass and 2nd-pass MiniMax H3 sampling.
- Shares cached head and tail latents across both passes for zero redundant VAE re-encoding.
- Injects optional **Bidirectional Tail Keyframes (`1 / 5 / 22f`)** with optional `reverse_taper` mild chroma noise (`alpha = 0.0` strictly at seam frame `pos = 0`) and 32 kHz tail audio anchoring for `22f`.
- Applies Fixed-Camera / Digital Human anti-degradation noise attenuation when configured in `ShotVaultPlan`.

---

**简体中文** | 面向 MiniMax H3 一采/二采的上下文窗口与双向续接引导节点。
- 一采/二采共享头尾潜变量缓存，零重复 VAE 编解码。
- 支持 **双头桥接尾锚 (`1 / 5 / 22f`)** 注入、`22f` 可选反向渐进轻彩噪（接缝 `pos = 0` 处严格 `alpha = 0.0` 纯净无噪）及 32kHz 尾音锚定。
- 响应分镜规划台的固定机位/数字人防劣化配置，自动调节彩噪强度与静态背景保护。
