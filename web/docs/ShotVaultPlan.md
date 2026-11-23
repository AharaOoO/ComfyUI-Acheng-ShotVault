# ShotVault Plan · 可视化分镜规划台 (v5.8)

**English** | Interactive storyboard & audio planner console for MiniMax H3 multi-shot production.
- **Tab ① Storyboard & Prompts**: Auto-split scripts (`& 8 &`, `@cut@`, `@continue@`, `===`), per-shot Action/Drama routing, 9-slot image refs (`<Picture 1..9>`), draggable height + full-screen Focus Prompt Editor (`Ctrl+Enter`).
- **Tab ② Reroll & Seam Studio**: Compare historical takes, run instant 3s seam repair preview (**🧪 Experimental**: only for sudden visual artifacts from continuation when camera angle & framing stay unchanged; usually leave OFF), and lock targeted **single-shot reroll** with optional **Bidirectional Tail Bridge (`0 / 1 / 5 / 22f`)** and **Auto-Refresh `master.mp4`**.
- **Tab ③ Shot 0 & Global Refs**: Upload a seed video to continue Shot #1 seamlessly, or configure a global 9-image fallback pool.
- **Tab ④ Gear · Anti-Deg & Checks**: Switch `22 / 39 / 56f` gears, configure Fixed-Camera / Digital Human Anti-Degradation modes, and verify preflight checks.
- **Header & Footer**: Live `EN / 中文` switcher (syncs all 10 canvas nodes + sidebar), manifest-verified `Export Film` modal, and clamped fluid panel presets (`Compact / Standard / Wide`).

---

**简体中文** | 面向 MiniMax H3 长视频分段制作的内嵌可视化分镜规划控制台。
- **① 分镜与提示词**：支持 `& 8 &` + `@cut@` / `@continue@` + `===` 自动分段、每镜独立武戏/文戏路由与 9 图槽位、可拖拽文本框与全屏专注编辑器（`Ctrl+Enter` 保存）。
- **② 选片重抽与修缝**：对比历史 Take、免重跑 H3 局部修缝预览（**🧪 实验功能**：仅适用于机位与分镜保持不变时，因续接生成导致的画面突变修补，平时基本无需开启）、单段定向重抽（支持 **尾接下一段纯净首部 `0 / 1 / 5 / 22f`** 与 **重抽后自动刷新主成片**）。
- **③ Shot 0 与全局图池**：上传已有视频续拍底板与全局 9 图回退池。
- **④ 档位 · 防劣化与体检**：`22 / 39 / 56` 帧联动矩阵、固定机位/数字人防劣化策略及硬性预检。
- **顶栏与底栏**：中英双语一键切换（联动画布全部 10 节点与侧边栏）、伴生 `.manifest.json` 过期防呆导出弹窗，以及 `紧凑 / 标准 / 宽屏` 尺寸预设。
