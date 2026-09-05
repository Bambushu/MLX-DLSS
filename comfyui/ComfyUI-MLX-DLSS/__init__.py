"""ComfyUI node pack for MLX-DLSS: the DLSS neural renderer and frame generator as nodes.

Install: symlink this folder into ComfyUI/custom_nodes and `pip install -e <repo>/python[mask]`
into ComfyUI's venv. Weights come from `mlxdlss-weights` (see the repository README).
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
