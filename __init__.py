"""ComfyUI-Acheng-ShotVault v5.8 entrypoint.

Copyright (C) 2026 AchengOoO (AharaOoO)
Licensed under the GNU General Public License v3.0 (GPL-3.0-only).
See LICENSE and NOTICE in the repository root for full terms.
"""

from __future__ import annotations

import logging

from .nodes import NODE_CLASS_MAPPINGS as _SV_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _SV_DISPLAY_MAPPINGS
from .server_routes import register_routes

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = dict(_SV_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(_SV_DISPLAY_MAPPINGS)

# Register bundled Acheng Seam Bridge & Clip Merge nodes so a single `git clone`
# provides the full 10-node suite out of the box.
try:
    from .seam_bridge import (
        NODE_CLASS_MAPPINGS as _SB_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _SB_DISPLAY_MAPPINGS,
    )

    NODE_CLASS_MAPPINGS.update(_SB_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_SB_DISPLAY_MAPPINGS)
except Exception as _sb_err:  # pragma: no cover
    logging.warning(f"[ComfyUI-Acheng-ShotVault] Optional seam_bridge nodes not loaded: {_sb_err}")

try:
    register_routes()
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
