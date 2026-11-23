# ShotVaultH3Conditioning [N2B]

动态按镜编码当前镜头在分镜规划面板上传的多模态参考素材（`<Picture 1..9>`、`<Video 1..3>`、`<Audio 1..3>`）以及画布额外直连参考信号，生成精准对齐 `MiniMax-H3` Packed-DiT 顺序的条件张量 `positive` 与 `latent`。

## 编码顺序与自动防坑特性
1. 按固定顺序编码：`全局共享参考图 + 本镜参考图`（`<Picture i>`） $\rightarrow$ `本镜参考视频`（若勾选带原声则自动前置 `<Audio j>` 再编 `<Video k>`） $\rightarrow$ `全局/本镜独立参考音色`（`<Audio m>`）。
2. 当某镜未上传任何素材时，保持干净零参考输入，**绝不编码全黑占位图造成模型污染**。
3. 支持随时将右侧模型、采样器、Sigma 与 LoRA 链替换为任意自定义节点，本节点仅输出标准的 `CONDITIONING` 与 `LATENT`。
