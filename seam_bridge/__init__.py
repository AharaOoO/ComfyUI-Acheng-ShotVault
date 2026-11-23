from .clip_merge_node import AchengFuzzPuppyClipMerge
from .seam_bridge import AchengSeamBridge

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "AchengSeamBridge": AchengSeamBridge,
    "AchengFuzzPuppyClipMerge": AchengFuzzPuppyClipMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AchengSeamBridge": "Acheng Seam Bridge · 接缝修复",
    "AchengFuzzPuppyClipMerge": "Acheng FuzzPuppy Clip Merge · 视频无缝合并",
}
