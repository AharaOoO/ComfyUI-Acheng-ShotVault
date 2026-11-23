# ShotVault Seam Trim (智能 ΔE 锁色 / 零帧差 RIFE 微桥)

`ShotVaultSeamTrim` 遵循第 0 端口 `IMAGE -> IMAGE` 同构对齐，支持 `Ctrl+B` 原生旁路，零额外整段视频重解码开销。

## 核心参数与工作模式
- **`trim_head_frames` / `keep_frames`**：默认均为 **`-1`**，自动从 `shot_pipe` 读取无漂移裁头与 `keep_frames` 净帧长。
- **`color_match_mode`**：
  - `auto_dual_mkl_lab` (**`v5.3` 默认**)：三道自动门控（同景双向块匹配门 + 爆闪过滤 + 静态背景掩膜 + CIE-Lab $\Delta E$ 死区 `color_deadzone_de=1.5`）驱动双层有界 MKL 调色：**全段锁定同场首镜锚点 (`global_lock_strength=0.75`) 阻断多段色偏滚雪球 + 接缝前 `14` 帧五次光滑释放**；无色差或硬切时 100% 原样直通。
  - `seam_12f_mkl_lab`：经典仅前 12 帧阶梯 MKL 消色差。
  - `masked_bg_ycc`：针对 `<Video 1>` 换人，仅按不动背景校准 YCbCr。
  - `single_video_auto_heal`：$O(1)$ 零拷贝单视频自动修缝模式（替代二次方全片重拷，自动按多段边界执行双层锁色 + 零帧差 RIFE 微桥）。
  - `off`：关闭校色（仍可按需触发 RIFE 微桥）。
- **`rife_bridge_mode` / `bridge_max_frames` / `jump_spike_threshold`**：
  - `auto_jump_rife` (**`v5.3` 默认**)：仅当接缝帧跳变比 $\ge 1.65$、瞬时帧差较大或出现首帧微冻结（且通过同景/非爆闪门控）时，使用本地 CUDA FP16 **`rife49.pth`**（缺权重时退回 OpenCV `DIS` 光流）合成 `2~4` 帧原位替换段首帧，**严格保持总帧数 0 增减、音频 0 裁剪**。
  - `always_rife`：同景续接每段固定生成 `bridge_max_frames` 帧 RIFE 微桥。
  - `off`：关闭 RIFE 微桥。
