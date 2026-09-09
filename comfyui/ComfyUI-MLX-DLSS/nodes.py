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
    # same resolution as the CLIs: explicit path, MLXDLSS_WEIGHTS / weights dir, then MLXDLSS_HF_REPO
    from mlxdlss.weights import resolve_weights

    return resolve_weights(text)


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def _progress_bar(total: int):
    """ComfyUI per-frame progress bar; None when comfy isn't importable (e.g. unit tests) or empty."""
    if total <= 0:
        return None
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(total)
    except (ImportError, AttributeError):
        return None


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
            "degrid": ("BOOLEAN", {"default": True, "tooltip": "notch the fine-tunes' period-4 token grid out of the residual (no effect on stock weights)"}),
        }, "optional": {
            "skin_mask": ("MASK", {"tooltip": "your own (batch, height, width) skin mask; overrides auto_mask"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, renderer, image, profile, processing_scale, detail_strength, colour_strength, detail_radius, intensity,
               auto_mask, mask_floor, mask_feather, noise_frame_index, degrid=True, skin_mask=None):
        masker = None
        if skin_mask is None and auto_mask == "skin":
            from mlxdlss.automask import SkinMasker

            key = ("masker", str(renderer.device), float(mask_feather))
            if key not in _cache:
                _cache[key] = SkinMasker(device=renderer.device, feather_sigma=float(mask_feather))
            masker = _cache[key]
        outputs, masks = [], []
        pbar = _progress_bar(image.shape[0])
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
                frame_index=int(noise_frame_index) + index, skin_mask=mask, degrid=bool(degrid),
            )
            outputs.append(torch.from_numpy(np.clip(result.image, 0, 1).astype(np.float32)))
            masks.append(torch.from_numpy((mask if mask is not None else np.zeros(frame.shape[:2], dtype=np.float32)).astype(np.float32)))
            if pbar is not None:
                pbar.update(1)
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


class MLXDLSSNeuralRenderingMetal:
    """Video-speed renderer through the Swift Metal runtime (macOS): ~17x the PyTorch path on
    temporal video, same output within 1/255. No skin auto-mask on this path (the Metal
    stream has no mask input yet)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "a frame sequence (temporal) or independent stills"}),
            "model_package": ("STRING", {"default": "NeuralRendering.dlssmodel", "tooltip": "from `mlxdlss-weights all`; relative paths resolve under MLXDLSS_WEIGHTS"}),
            "mlxdlss_binary": ("STRING", {"default": "", "tooltip": "path to the built `mlxdlss` binary; empty = MLXDLSS_BINARY / PATH / the repo's .build/release"}),
            "temporal": ("BOOLEAN", {"default": True, "tooltip": "reproject the previous output with optical flow and blend (video); off = every frame independent"}),
            "scene_cut": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "temporal: mean luma change that resets the history"}),
            "profile": (list(PROFILES), {"default": "standard"}),
            "processing_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.5, "tooltip": "temporal mode runs at the native scale (1)"}),
            "detail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1}),
            "colour_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1}),
            "detail_radius": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 32.0, "step": 0.5}),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "precision": (["float16", "float32"], {"default": "float16"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "scene_cuts")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, images, model_package, mlxdlss_binary, temporal, scene_cut, profile, processing_scale, detail_strength,
               colour_strength, detail_radius, intensity, precision):
        from mlxdlss.mlxdlss_stream import MLXDLSSStreamSession

        if temporal and float(processing_scale) != 1.0:
            raise ValueError("temporal mode runs at the native scale: set processing_scale to 1 or temporal off")
        height, width = int(images.shape[1]), int(images.shape[2])
        session = MLXDLSSStreamSession(
            _path(model_package), width, height, temporal=bool(temporal), scene_cut_threshold=float(scene_cut),
            mlxdlss=mlxdlss_binary or None, profile=profile, intensity=float(intensity), precision=precision,
            processing_scale=float(processing_scale), detail_strength=float(detail_strength),
            colour_strength=float(colour_strength), detail_radius=float(detail_radius),
        )
        try:
            outputs = [torch.from_numpy(np.clip(session.process_frame(images[i].detach().float().clamp(0, 1).cpu().numpy()), 0, 1).astype(np.float32))
                       for i in range(images.shape[0])]
            cuts = int(getattr(session, "scene_cuts", 0))
        finally:
            session.close()
        return (torch.stack(outputs), cuts)


# ----------------------------------------------------------------------------- upscale (resample or model, then neural re-detail)

RESAMPLERS = {"lanczos": "LANCZOS", "bicubic": "BICUBIC"}


def _resample(frame: np.ndarray, size: tuple[int, int], method: str) -> np.ndarray:
    from PIL import Image

    resample = getattr(Image, RESAMPLERS[method])
    return np.asarray(Image.fromarray(_to_uint8(torch.from_numpy(frame))).resize(size, resample)).astype(np.float32) / 255.0


def _upscale(frame: np.ndarray, size: tuple[int, int], method: str, upscale_model, device) -> np.ndarray:
    """(H, W, 3) float in [0, 1] -> (size[1], size[0], 3). ``upscale_model`` is ComfyUI's UPSCALE_MODEL (spandrel);
    its fixed 2x/4x output is Lanczos-resampled to the exact target."""
    if method == "upscale_model":
        if upscale_model is None:
            raise ValueError("method upscale_model needs an UPSCALE_MODEL input (ComfyUI's Load Upscale Model)")
        model = upscale_model.to(device) if hasattr(upscale_model, "to") else upscale_model
        with torch.no_grad():
            big = model(torch.from_numpy(frame).permute(2, 0, 1)[None].to(device)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return big if big.shape[:2] == (size[1], size[0]) else _resample(big, size, "lanczos")
    return _resample(frame, size, method)


def _masker(renderer, auto_mask: str, mask_feather: float):
    if auto_mask != "skin":
        return None
    from mlxdlss.automask import SkinMasker

    key = ("masker", str(renderer.device), float(mask_feather))
    if key not in _cache:
        _cache[key] = SkinMasker(device=renderer.device, feather_sigma=float(mask_feather))
    return _cache[key]


UPSCALE_INPUTS = {
    "scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25, "tooltip": "2 = most visible re-detail (default); 1.5 is faster with less base softening. Pixels come from the resampler/model, detail from the renderer"}),
    "method": (["lanczos", "bicubic", "upscale_model"], {"default": "lanczos", "tooltip": "upscale_model: plug ComfyUI's Load Upscale Model (SPAN/ESRGAN...) into upscale_model"}),
    "detail_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": "detail-enhancement pass on soft/upscaled video: adds mostly-synthesised fine texture (only ~30% aligns with the true lost detail, measured — see ab/RESULTS.md), so enhancement, not reconstruction. Does not invent objects or change the subject. 1 = subtle, 2 = visible crispness (default), 3 = grain"}),
    "colour_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1, "tooltip": "fine-tuned weights: 1; stock weights: 0.5"}),
    "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
    "auto_mask": (["skin", "none"], {"default": "skin"}),
    "mask_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
    "mask_feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 1.0}),
    "noise_frame_index": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
    "degrid": ("BOOLEAN", {"default": True, "tooltip": "notch the fine-tunes' period-4 token grid out of the residual"}),
}


class MLXDLSSImageUpscale:
    """Image (batch) upscale: resampler or upscale model for the pixels, then the neural renderer at processing
    scale 1 for the detail. Defaults = the fine-tune recipe (use the fine-tuned weights in the loader)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"renderer": ("MLXDLSS_RENDERER",), "image": ("IMAGE",), **UPSCALE_INPUTS},
                "optional": {"upscale_model": ("UPSCALE_MODEL",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, renderer, image, scale_factor, method, detail_strength, colour_strength, intensity, auto_mask, mask_floor, mask_feather,
                noise_frame_index, degrid=True, upscale_model=None):
        masker = _masker(renderer, auto_mask, mask_feather)
        size = (round(image.shape[2] * float(scale_factor)), round(image.shape[1] * float(scale_factor)))
        outputs = []
        pbar = _progress_bar(image.shape[0])
        for index in range(image.shape[0]):
            frame = _upscale(image[index].detach().float().clamp(0, 1).cpu().numpy(), size, method, upscale_model, renderer.device)
            mask = masker.mask(frame, floor=float(mask_floor)) if masker is not None else None
            result = renderer.enhance(frame, profile="standard", processing_scale=1.0, detail_strength=float(detail_strength), colour_strength=float(colour_strength),
                                      intensity=float(intensity), frame_index=int(noise_frame_index) + index, skin_mask=mask, degrid=bool(degrid))
            outputs.append(torch.from_numpy(np.clip(result.image, 0, 1).astype(np.float32)))
            if pbar is not None:
                pbar.update(1)
        return (torch.stack(outputs),)


class MLXDLSSVideoUpscale:
    """Video upscale: the image path per frame plus the temporal session (optical-flow history, gated high-pass
    history blend against shimmer, scene-cut reset). Needs `pip install 'mlxdlss[video]'` for the flow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"renderer": ("MLXDLSS_RENDERER",), "image": ("IMAGE",), **UPSCALE_INPUTS,
                             "hp_history": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.05, "tooltip": "history weight on the high-pass band: 0.5-0.7 removes static shimmer for ~5% detail"}),
                             "scene_cut": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "mean luma change that resets the history (0 = never)"})},
                "optional": {"upscale_model": ("UPSCALE_MODEL",)}}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "scene_cuts")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, renderer, image, scale_factor, method, detail_strength, colour_strength, intensity, auto_mask, mask_floor, mask_feather,
                noise_frame_index, degrid, hp_history, scene_cut, upscale_model=None):
        from mlxdlss.temporal import TemporalOptions, TemporalSession

        masker = _masker(renderer, auto_mask, mask_feather)
        size = (round(image.shape[2] * float(scale_factor)), round(image.shape[1] * float(scale_factor)))
        session = TemporalSession(renderer, options=TemporalOptions(
            profile="standard", detail_strength=float(detail_strength), colour_strength=float(colour_strength), intensity=float(intensity),
            scene_cut_threshold=float(scene_cut) if scene_cut > 0 else 2.0, hp_history=float(hp_history), degrid=bool(degrid)))
        outputs = []
        pbar = _progress_bar(image.shape[0])
        for index in range(image.shape[0]):
            frame = _upscale(image[index].detach().float().clamp(0, 1).cpu().numpy(), size, method, upscale_model, renderer.device)
            mask = masker.mask(frame, floor=float(mask_floor)) if masker is not None else None
            outputs.append(torch.from_numpy(np.clip(session.process(frame, skin_mask=mask), 0, 1).astype(np.float32)))
            if pbar is not None:
                pbar.update(1)
        return (torch.stack(outputs), int(session.scene_cuts))


NODE_CLASS_MAPPINGS = {
    "MLXDLSSLoadRenderer": MLXDLSSLoadRenderer,
    "MLXDLSSNeuralRendering": MLXDLSSNeuralRendering,
    "MLXDLSSNeuralRenderingMetal": MLXDLSSNeuralRenderingMetal,
    "MLXDLSSImageUpscale": MLXDLSSImageUpscale,
    "MLXDLSSVideoUpscale": MLXDLSSVideoUpscale,
    "MLXDLSSLoadFrameGen": MLXDLSSLoadFrameGen,
    "MLXDLSSFrameGeneration": MLXDLSSFrameGeneration,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXDLSSLoadRenderer": "MLX-DLSS Load Neural Renderer",
    "MLXDLSSNeuralRendering": "MLX-DLSS Neural Rendering (detail + tone)",
    "MLXDLSSNeuralRenderingMetal": "MLX-DLSS Neural Rendering VIDEO (Metal, temporal)",
    "MLXDLSSImageUpscale": "DLSSDetailer — Image Upscale (resample + re-detail)",
    "MLXDLSSVideoUpscale": "DLSSDetailer — Video Upscale (resample + re-detail, temporal)",
    "MLXDLSSLoadFrameGen": "MLX-DLSS Load Frame Generator",
    "MLXDLSSFrameGeneration": "MLX-DLSS Frame Generation (interpolate)",
}
