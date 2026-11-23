# ShotVault Loop Dispatch (`ShotVaultLoopDispatch`)

`[N2D]` 是专为外部循环架构（如 `easy forLoopStart` 或自定义循环/重抽工作流）设计的**无状态单镜分发器**——本身**绝不内置队列循环**，彻底把“如何循环”的决定权交给画布外部节点。

## 核心作用
1. **外部索引路由**：读取 `0-based (easy forLoopStart)` 或 `1-based` 段号，精确分发当前第 `k` 镜的 `prompt`、`is_drama_mode`（武戏 / 文戏）、`target_frames`、`keep_frames`、`seed`。
2. **`22 / 39` 全链路完全换挡不漏项**：同步输出 `overlap_frames`、`trim_head_frames`、`taper_tail_frames`、`taper_ramp_frames`、`audio_guide_frames`（`20f / 37f`，严格留 `2f` 防爆音带）与 `audio_feather_ticks`（`7 / 12`）。首段非续接时自动保证 `taper_tail_frames >= 22`，防止 `#749` 报错。
3. **每段独立 9 张参考图输出 (`ref_image_1 .. ref_image_9`)**：自动按 BFS 9 槽顺序继承全局池与本段覆盖槽位，空槽返回 `None`，绝不塞入黑图占位。
4. **响应分镜规划重抽锁**：当分镜规划面板启用单段重抽锁时，自动定向该段索引、覆盖种子与重试提示词。
