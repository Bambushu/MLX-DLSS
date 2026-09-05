"""MLX-DLSS nodes. IMAGE tensors are ComfyUI's (batch, height, width, 3) float32 in [0, 1]."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from mlxdlss.features import PROFILES
from mlxdlss.framegen import FrameGenerator
from mlxdlss.framegen_video import is_scene_cut
from mlxdlss.pipeline import NeuralRenderingPipeline

CATEGORY = "MLX-DLSS"
DEFAULT_WEIGHTS = os.environ.get("MLXDLSS_WEIGHTS", str(Path("~/mlx-dlss/weights").expanduser()))
_cache: dict[tuple, object] = {}


def _path(text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(DEFAULT_WEIGHTS) / path
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path}")
    return path


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


class MLXDLSSLoadRenderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "weights": ("STRING", {"default": "dlssnr-weights-logical.safetensors", "tooltip": "logical safetensors from `mlxdlss-weights all`; relative paths resolve under MLXDLSS_WEIGHTS (~/mlx-dlss/weights)"}),
            "device": (["auto", "mps", "cuda", "cpu"], {"default": "auto"}),
            "precision": (["fast", "reference"], {"default": "fast", "tooltip": "fast = half precision on the GPU; reference = the bit-matched graph"}),
        }}

    RETURN_TYPES = ("MLXDLSS_RENDERER",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, weights, device, precision):
        key = ("renderer", str(_path(weights)), device, precision)
        if key not in _cache:
            _cache[key] = NeuralRenderingPipeline.from_safetensors(_path(weights), device=device, precision=precision)
        return (_cache[key],)


class MLXDLSSNeuralRendering:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "renderer": ("MLXDLSS_RENDERER",),
            "image": ("IMAGE",),
            "profile": (list(PROFILES), {"default": "standard"}),
            "processing_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5, "tooltip": "run the network on the frame resampled by this factor (2 = the Krea2 still recipe; cost grows with its square)"}),
            "detail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1}),
            "colour_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1, "tooltip": "1 = vendor default, darkens/desaturates warm scenes ~5%; 0.3-0.5 keeps the light"}),
            "detail_radius": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 32.0, "step": 0.5}),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "auto_mask": (["none", "skin"], {"default": "skin", "tooltip": "skin: face-parsing model confines the skin detail to face skin (needs transformers)"}),
            "mask_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "skin-channel value outside the detected face (chest/arms are not detected)"}),
            "mask_feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 1.0}),
            "noise_frame_index": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "tooltip": "deterministic noise seed; advanced per frame for batches"}),
        }, "optional": {
            "skin_mask": ("MASK", {"tooltip": "your own (batch, height, width) skin mask; overrides auto_mask"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, renderer, image, profile, processing_scale, detail_strength, colour_strength, detail_radius, intensity,
               auto_mask, mask_floor, mask_feather, noise_frame_index, skin_mask=None):
        masker = None
        if skin_mask is None and auto_mask == "skin":
            from mlxdlss.automask import SkinMasker

            key = ("masker", str(renderer.device), float(mask_feather))
            if key not in _cache:
                _cache[key] = SkinMasker(device=renderer.device, feather_sigma=float(mask_feather))
            masker = _cache[key]
        outputs, masks = [], []
        for index in range(image.shape[0]):
            frame = image[index].detach().float().clamp(0, 1).cpu().numpy()
            if skin_mask is not None:
                mask = skin_mask[min(index, skin_mask.shape[0] - 1)].detach().float().clamp(0, 1).cpu().numpy()
                if mask.shape != frame.shape[:2]:
                    mask = torch.nn.functional.interpolate(torch.from_numpy(mask)[None, None], size=frame.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
            elif masker is not None:
                mask = masker.mask(frame, floor=float(mask_floor))
            else:
                mask = None
            result = renderer.enhance(
                frame, profile=profile, processing_scale=float(processing_scale), detail_strength=float(detail_strength),
                colour_strength=float(colour_strength), detail_radius=float(detail_radius), intensity=float(intensity),
                frame_index=int(noise_frame_index) + index, skin_mask=mask,
            )
            outputs.append(torch.from_numpy(np.clip(result.image, 0, 1).astype(np.float32)))
            masks.append(torch.from_numpy((mask if mask is not None else np.zeros(frame.shape[:2], dtype=np.float32)).astype(np.float32)))
        return (torch.stack(outputs), torch.stack(masks))


class MLXDLSSLoadFrameGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "weights": ("STRING", {"default": "framegen.safetensors", "tooltip": "dense safetensors from `mlxdlss-weights extract-fg`"}),
            "device": (["auto", "mps", "cuda", "cpu"], {"default": "auto"}),
            "precision": (["fast", "reference"], {"default": "fast"}),
        }}

    RETURN_TYPES = ("MLXDLSS_FRAMEGEN",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, weights, device, precision):
        key = ("framegen", str(_path(weights)), device, precision)
        if key not in _cache:
            _cache[key] = FrameGenerator.from_safetensors(_path(weights), device=device, precision=precision)
        return (_cache[key],)


class MLXDLSSFrameGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "framegen": ("MLXDLSS_FRAMEGEN",),
            "images": ("IMAGE", {"tooltip": "consecutive frames; factor-1 frames are generated between each pair"}),
            "factor": ("INT", {"default": 2, "min": 2, "max": 8, "tooltip": "2 doubles the frame count (fps x2 or 2x slowmo, your choice downstream)"}),
            "scene_cut": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean luma change between a pair that marks a cut: held as a hard cut instead of blended; 0 disables"}),
            "batch": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "pairs per network pass"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "scene_cuts")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, framegen, images, factor, scene_cut, batch):
        frames = [_to_uint8(images[i]) for i in range(images.shape[0])]
        if len(frames) < 2:
            return (images, 0)
        per_pair = factor - 1
        hold_first = [k / factor < 0.5 for k in range(1, factor)]
        out: list[np.ndarray] = [frames[0]]
        cuts = 0
        for start in range(0, len(frames) - 1, batch):
            window = frames[start:start + batch + 1]
            generated = framegen.generate_pairs(window, factor)
            for i, pair in enumerate(generated):
                a, b = window[i], window[i + 1]
                if is_scene_cut(a, b, float(scene_cut)):
                    cuts += 1
                    pair = [a if hold_first[k] else b for k in range(per_pair)]
                out.extend(pair)
                out.append(b)
        stacked = torch.from_numpy(np.stack(out).astype(np.float32) / 255.0)
        return (stacked, cuts)


NODE_CLASS_MAPPINGS = {
    "MLXDLSSLoadRenderer": MLXDLSSLoadRenderer,
    "MLXDLSSNeuralRendering": MLXDLSSNeuralRendering,
    "MLXDLSSLoadFrameGen": MLXDLSSLoadFrameGen,
    "MLXDLSSFrameGeneration": MLXDLSSFrameGeneration,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXDLSSLoadRenderer": "MLX-DLSS Load Neural Renderer",
    "MLXDLSSNeuralRendering": "MLX-DLSS Neural Rendering (detail + tone)",
    "MLXDLSSLoadFrameGen": "MLX-DLSS Load Frame Generator",
    "MLXDLSSFrameGeneration": "MLX-DLSS Frame Generation (interpolate)",
}
