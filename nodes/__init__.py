"""Node package exports for ComfyUI-Universal-ShotVault (8 orthogonal nodes)."""

from .assemble_node import ShotVaultAssemble
from .h3_conditioning_node import ShotVaultH3Conditioning
from .load_node import ShotVaultLoad
from .loop_dispatch_node import ShotVaultLoopDispatch
from .plan_node import ShotVaultPlan
from .save_node import ShotVaultSave
from .seam_trim_node import ShotVaultSeamTrim
from .window_guider_node import ShotVaultWindowGuider

NODE_CLASS_MAPPINGS = {
    "ShotVaultPlan": ShotVaultPlan,
    "ShotVaultLoad": ShotVaultLoad,
    "ShotVaultLoopDispatch": ShotVaultLoopDispatch,
    "ShotVaultH3Conditioning": ShotVaultH3Conditioning,
    "ShotVaultWindowGuider": ShotVaultWindowGuider,
    "ShotVaultSeamTrim": ShotVaultSeamTrim,
    "ShotVaultSave": ShotVaultSave,
    "ShotVaultAssemble": ShotVaultAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotVaultPlan": "ShotVault Plan (可视化分镜规划/音轨编排)",
    "ShotVaultLoad": "ShotVault Load (选段/双尾与多模加载)",
    "ShotVaultLoopDispatch": "ShotVault Loop Dispatch (外部循环分发/22·39档/9图)",
    "ShotVaultH3Conditioning": "ShotVault H3 Conditioning (分镜多模参照编码)",
    "ShotVaultWindowGuider": "ShotVault Window Guider (零劣化窗口引导)",
    "ShotVaultSeamTrim": "ShotVault Seam Trim (智能ΔE锁色/RIFE微桥)",
    "ShotVaultSave": "ShotVault Save (双轨隔离归档)",
    "ShotVaultAssemble": "ShotVault Assemble (单遍PCM总装)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
