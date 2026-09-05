"""Automatic skin/face control masks for the neural renderer.

The vendor fills the network's skin and automatic-mask channels from its own segmentation,
which is not part of the port. This module stands in with a face-parsing SegFormer
(``jonathandinu/face-parsing``, 19 CelebAMask-HQ classes) and turns the result into the
per-pixel skin mask the pipeline writes to feature channels 13 and 14 (``skin_mask``). Tone
and structure stay global; the network's skin-specific detail is confined to skin, with an
optional floor elsewhere. (Routing the mask through the control mask's blue channel instead
zeroes channels 13/14 and switches the skin detail off — measured, see ab/RESULTS.md.)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .composition import blur, gaussian_kernel

MODEL_ID = "jonathandinu/face-parsing"
# CelebAMask-HQ label ids of the face-parsing model that count as skin
SKIN_LABELS = {1: "skin", 2: "nose", 4: "l_eye", 5: "r_eye", 6: "l_brow", 7: "r_brow", 8: "l_ear", 9: "r_ear",
               10: "mouth", 11: "u_lip", 12: "l_lip", 17: "neck"}
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NETWORK_SIZE = 512


def feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-soften a (height, width) mask in [0, 1]; ``sigma <= 0`` returns it unchanged."""
    mask = np.asarray(mask, dtype=np.float32)
    if sigma <= 0:
        return mask
    kernel = gaussian_kernel(sigma)
    return np.clip(blur(blur(mask, kernel).T, kernel).T, 0, 1).astype(np.float32)


def skin_mask_with_floor(skin: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """``floor + (1 - floor) * skin`` for a (height, width) skin mask in [0, 1]: the value the
    network's skin channels get outside the face."""
    skin = np.asarray(skin, dtype=np.float32)
    if skin.ndim != 2:
        raise ValueError("skin mask must be (height, width)")
    if not 0 <= floor <= 1:
        raise ValueError("floor must be within [0, 1]")
    return (np.float32(floor) + np.float32(1 - floor) * np.clip(skin, 0, 1)).astype(np.float32)


class SkinMasker:
    """Face-parsing segmenter returning a soft skin mask per image (lazy model load)."""

    def __init__(self, device: str = "auto", feather_sigma: float = 8.0, model_id: str = MODEL_ID):
        self.device_name = device
        self.feather_sigma = feather_sigma
        self.model_id = model_id
        self._model: Any = None

    def _load(self) -> None:
        import torch

        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("the automatic skin mask needs transformers: pip install './python[mask]'") from error
        from .pipeline import resolve_device

        self.device = resolve_device(self.device_name)
        self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_id).eval().to(self.device)
        self._torch = torch

    def labels(self, image: np.ndarray) -> np.ndarray:
        """Per-pixel class ids (height, width) int64 for a (height, width, 3) float32 image in [0, 1]."""
        if self._model is None:
            self._load()
        torch = self._torch
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be (height, width, 3)")
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
        x = torch.nn.functional.interpolate(x, size=(NETWORK_SIZE, NETWORK_SIZE), mode="bilinear", align_corners=False)
        x = (x - torch.from_numpy(IMAGE_MEAN).view(1, 3, 1, 1)) / torch.from_numpy(IMAGE_STD).view(1, 3, 1, 1)
        with torch.inference_mode():
            logits = self._model(pixel_values=x.to(self.device)).logits
            logits = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
            return logits.argmax(1)[0].cpu().numpy()

    def skin(self, image: np.ndarray) -> np.ndarray:
        """Soft (height, width) skin mask in [0, 1]: skin-class pixels, feathered."""
        labels = self.labels(image)
        hard = np.isin(labels, list(SKIN_LABELS)).astype(np.float32)
        return feather(hard, self.feather_sigma)

    def mask(self, image: np.ndarray, floor: float = 0.0) -> np.ndarray:
        """The pipeline's ``skin_mask`` for ``image`` (see ``skin_mask_with_floor``)."""
        return skin_mask_with_floor(self.skin(image), floor)
