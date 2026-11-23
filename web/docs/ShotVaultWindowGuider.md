# ShotVault Window Guider (零劣化窗口引导)

`ShotVaultWindowGuider` 融合单镜 `ContextWindows` 步内 $x_0$ 锚定、`1st+2nd (22+5)` 非对称引导、`keyframe_until_sigma` 低噪放锁，以及二采局部 `sigma` 音频重定基保护。

## 关键参数
- **`pass_stage`**: 一采选 `single_or_1st`；二采精修选 `2nd_refine`。
- **`window_mode`**: 当目标长度超过 `window_frames`（`192f ≈ 8s`）自动启用步内 $x_0$ 窗口锚定 + 末 2 token `_late_handover`；含 `<d>` 对白时会自动结合 `+trim_head_frames` 偏移后的时间段执行 DP 避让排窗。
- **`audio_feather_ticks`**: 默认 `7`，可在节点上自由调节前序接续音频尾部的半余弦羽化长度（`0` 为硬接）。
- **`rebase_2nd_pass_audio`**: 默认开启，当底层使用非原生缩放（`audio_scale == 1.0`）进行二采部分去噪重启时自动将音频流从视频时间轴重定基到 `shift=3.0` 音频时钟；原生 `ModelSamplingAV` 则自动跳过防双重缩放。
- **单窗无负向条件**: 当 `negative` 未接且 `cfg=1.0` 时严格走原生 `Guider_Basic` 轻量路径，零 `None` 报错。
