"""Temporal path: reprojected display history in the network input and a learned blend.

Recovered contract (see docs/recovery-notes.md): the previous output is sampled
at ``currentUV + motion`` with a five-tap Catmull-Rom filter, scaled like the
colour channels and written into feature channels 7-9; the head's fourth
channel is a sigmoid blend logit capped by ``blend_scale`` that mixes the
predicted RGB with that history. Motion is a normalised history-UV offset:
positive values move the sample right/down.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .composition import compose_detail, compose_head
from .features import PROFILES, AutomaticMask, NetworkGeometry, deterministic_noise, half, make_features, scaled_color

BLEND_SCALE = 0.73974609375  # half(blend_scale) of the recovered package


def normalize_pixel_motion(
    pixel_motion: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    effective_width: int,
    effective_height: int,
    jitter_dx: float = 0.0,
    jitter_dy: float = 0.0,
) -> np.ndarray:
    """Engine pixel motion (H, W, 2) -> normalised history-UV offsets, with an optional previous-minus-current jitter delta."""
    pixel_motion = np.asarray(pixel_motion, dtype=np.float32)
    if pixel_motion.ndim != 3 or pixel_motion.shape[2] != 2:
        raise ValueError("pixel motion must be (height, width, 2)")
    if effective_width <= 0 or effective_height <= 0:
        raise ValueError("effective extent must be positive")
    for value in (scale_x, scale_y, jitter_dx, jitter_dy):
        if not np.isfinite(value):
            raise ValueError("motion scale and jitter must be finite")
    out = np.empty_like(pixel_motion)
    out[..., 0] = pixel_motion[..., 0] * np.float32(scale_x / effective_width) + np.float32(jitter_dx / effective_width)
    out[..., 1] = pixel_motion[..., 1] * np.float32(scale_y / effective_height) + np.float32(jitter_dy / effective_height)
    return out


def _catmull_coordinates(normalized: np.ndarray, dimension: int):
    pixel = normalized * np.float32(dimension) - np.float32(0.5)
    base_index = np.floor(pixel)
    t = np.clip(pixel - base_index, 0, 1).astype(np.float32)
    square = t * t; cube = square * t
    w0 = -0.5 * t + square - 0.5 * cube
    w1 = 1 - 2.5 * square + 1.5 * cube
    w2 = 0.5 * t + 2 * square - 1.5 * cube
    w3 = -0.5 * square + 0.5 * cube
    g = w1 + w2
    base = base_index + np.float32(0.5)
    lower, upper = np.float32(0.5), np.float32(dimension) - np.float32(0.5)
    return (
        np.clip(base - 1, lower, upper), np.clip(base + w2 / g, lower, upper), np.clip(base + 2, lower, upper),
        w0.astype(np.float32), w3.astype(np.float32), g.astype(np.float32),
    )


def _sample_linear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of (H, W, C) at pixel-centre coordinates (x, y), clamped to the edge."""
    height, width = image.shape[:2]
    px = x - np.float32(0.5); py = y - np.float32(0.5)
    x0 = np.clip(np.floor(px), 0, width - 1).astype(np.int64); y0 = np.clip(np.floor(py), 0, height - 1).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1); y1 = np.minimum(y0 + 1, height - 1)
    tx = np.clip(px - x0, 0, 1).astype(np.float32)[..., None]; ty = np.clip(py - y0, 0, 1).astype(np.float32)[..., None]
    top = image[y0, x0] * (1 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1 - tx) + image[y1, x1] * tx
    return top * (1 - ty) + bottom * ty


def sample_history(history: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Five-tap Catmull-Rom approximation (cross of bilinear taps) at normalised coordinates (u, v)."""
    history = np.asarray(history, dtype=np.float32)
    height, width = history.shape[:2]
    x_outer0, x_middle, x_outer3, x_w0, x_w3, x_g = _catmull_coordinates(np.asarray(u, dtype=np.float32), width)
    y_outer0, y_middle, y_outer3, y_w0, y_w3, y_g = _catmull_coordinates(np.asarray(v, dtype=np.float32), height)
    weights = [x_w0 * y_g, x_g * y_w0, x_g * y_g, x_g * y_w3, x_w3 * y_g]
    taps = [
        _sample_linear(history, x_outer0, y_middle), _sample_linear(history, x_middle, y_outer0),
        _sample_linear(history, x_middle, y_middle), _sample_linear(history, x_middle, y_outer3),
        _sample_linear(history, x_outer3, y_middle),
    ]
    total = sum(w[..., None] * tap for w, tap in zip(weights, taps))
    return (total / sum(weights)[..., None]).astype(np.float32)


def _closest_depth_offsets(depth: np.ndarray, inverted: bool):
    """Per-pixel source coordinate of the closest of the pixel and its four diagonals (dormant vendor branch)."""
    height, width = depth.shape[:2]
    yy, xx = np.indices((height, width))
    best_x, best_y, best = xx.copy(), yy.copy(), depth[..., 0].copy()
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cx = np.clip(xx + dx, 0, width - 1); cy = np.clip(yy + dy, 0, height - 1)
        candidate = depth[cy, cx, 0]
        closer = candidate > best if inverted else candidate < best
        best_x = np.where(closer, cx, best_x); best_y = np.where(closer, cy, best_y); best = np.where(closer, candidate, best)
    return best_x, best_y


def make_temporal_features(
    color: np.ndarray,
    history: np.ndarray,
    motion: np.ndarray,
    *,
    frame_index: int,
    depth: np.ndarray | None = None,
    depth_guide: str = "observed",
    depth_inverted: bool = False,
    normalized_style: float = 0.0,
    local_tone_strength: float = 1.0,
    local_structure_strength: float = 1.0,
    automatic_mask: AutomaticMask | None = None,
    control_mask: np.ndarray | None = None,
    skin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Logical-size (H, W, 16) features: first-frame layout with reprojected history in channels 7-9."""
    color = np.asarray(color, dtype=np.float32); history = np.asarray(history, dtype=np.float32); motion = np.asarray(motion, dtype=np.float32)
    height, width = color.shape[:2]
    if history.shape != color.shape:
        raise ValueError("history must match the colour shape")
    if motion.shape != (height, width, 2):
        raise ValueError("motion must be (height, width, 2)")
    features = make_features(
        color, frame_index=frame_index, normalized_style=normalized_style, local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength, automatic_mask=automatic_mask, control_mask=control_mask, skin_mask=skin_mask,
    )
    yy, xx = np.indices((height, width))
    if depth_guide == "closest":
        if depth is None:
            raise ValueError("closest-depth guide needs a depth map")
        sx, sy = _closest_depth_offsets(np.asarray(depth, dtype=np.float32).reshape(height, width, -1), depth_inverted)
        sampled_motion = motion[sy, sx]
    elif depth_guide == "observed":
        sampled_motion = motion
    else:
        raise ValueError("depth_guide must be 'observed' or 'closest'")
    u = (xx.astype(np.float32) + np.float32(0.5)) / np.float32(width) + sampled_motion[..., 0]
    v = (yy.astype(np.float32) + np.float32(0.5)) / np.float32(height) + sampled_motion[..., 1]
    features[..., 7:10] = scaled_color(sample_history(history, u, v))
    return features


def extend_features(features: np.ndarray, geometry: NetworkGeometry, frame_index: int) -> np.ndarray:
    """Mirror logical features onto the network extent and regenerate the noise in the extension."""
    if geometry.is_identity:
        return features
    rows, columns = geometry.source_rows(), geometry.source_columns()
    extended = features[rows[:, None], columns[None, :], :].copy()
    noise = deterministic_noise(geometry.network_height, geometry.network_width, frame_index)
    outside = (rows[:, None] != np.arange(geometry.network_height)[:, None]) | (columns[None, :] != np.arange(geometry.network_width)[None, :])
    extended[..., 0:3] = np.where(outside[..., None], noise, extended[..., 0:3])
    return extended


def compose_temporal(
    head: np.ndarray,
    color: np.ndarray,
    features: np.ndarray,
    *,
    blend_scale: float = BLEND_SCALE,
    control_mask: np.ndarray | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    """predicted + alpha * (history - predicted), alpha = clamp(sigmoid(half(logit)) * half(blend_scale))."""
    head = np.asarray(head, dtype=np.float32); color = np.asarray(color, dtype=np.float32); features = np.asarray(features, dtype=np.float32)
    if head.shape[:2] != color.shape[:2] or features.shape[:2] != color.shape[:2] or features.shape[2] != 16:
        raise ValueError("head, colour and features must share height and width; features need 16 channels")
    logit = half(head[..., 3:4])
    alpha = np.clip(1 / (1 + np.exp(-logit)) * half(blend_scale), 0, 1)
    predicted = np.clip(color + half(head[..., :3]) * np.float32(0.25), 0, 1)
    history = features[..., 7:10] * np.float32(8) + np.float32(0.5)
    temporal = predicted + alpha * (history - predicted)
    if control_mask is None and intensity == 1:
        return temporal.astype(np.float32)
    blend = np.float32(intensity) if control_mask is None else np.asarray(control_mask, dtype=np.float32)[..., :1] * np.float32(intensity)
    blend = np.clip(blend, 0, 1)
    return np.clip(color + blend * (temporal - color), 0, 1).astype(np.float32)


class FlowMotionEstimator:
    """Dense optical flow (OpenCV DIS) from the current frame to the previous one, as normalised history-UV offsets."""

    def __init__(self, preset: str = "medium"):
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("optical-flow motion needs OpenCV: pip install 'mlxdlss[video]'") from error
        presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
        self._cv2 = cv2
        self._flow = cv2.DISOpticalFlow_create(presets[preset])

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        luma = frame[..., 0] * 0.2126 + frame[..., 1] * 0.7152 + frame[..., 2] * 0.0722
        return (np.clip(luma, 0, 1) * 255 + 0.5).astype(np.uint8)

    def __call__(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        flow = self._flow.calc(self._gray(current), self._gray(previous), None)  # current -> previous, pixels
        height, width = current.shape[:2]
        return normalize_pixel_motion(flow, scale_x=1, scale_y=1, effective_width=width, effective_height=height)


def zero_motion(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return np.zeros((*current.shape[:2], 2), dtype=np.float32)


@dataclass
class TemporalOptions:
    profile: str = "standard"
    blend_scale: float = BLEND_SCALE
    intensity: float = 1.0
    scene_cut_threshold: float = 0.3
    detail_strength: float = 1.0
    colour_strength: float = 1.0
    detail_radius: float = 4.0
    normalized_style: float | None = None
    local_tone_strength: float | None = None
    local_structure_strength: float | None = None
    noise_mode: str = "fresh"       # fresh (vendor: new noise per frame) | frozen | zero | advected (noise field follows the motion)
    hp_history: float = 0.0         # extra history weight on the HIGH-PASS band only (0 = vendor blend), gated by photometric warp error
    hp_history_sigma: float = 2.0   # gaussian sigma of the band split


class TemporalSession:
    """Frame-sequence processor with display history, motion reprojection and the learned blend.

    ``motion`` is a callable ``(current, previous) -> (H, W, 2)`` normalised offsets (default: optical flow),
    or engine motion passed per frame to ``process``. A scene cut (mean absolute luma change above the
    threshold) or ``reset`` clears the history and restarts the noise frame index, like the Swift backend.
    """

    def __init__(self, pipeline, *, options: TemporalOptions | None = None, motion: Callable | str = "flow"):
        self.pipeline = pipeline
        self.options = options or TemporalOptions()
        if self.options.profile not in PROFILES:
            raise ValueError(f"profile must be one of {tuple(PROFILES)}")
        if motion == "flow":
            self.motion = FlowMotionEstimator()
        elif motion == "zero":
            self.motion = zero_motion
        elif callable(motion):
            self.motion = motion
        else:
            raise ValueError("motion must be 'flow', 'zero' or a callable")
        self.history: np.ndarray | None = None
        self.previous: np.ndarray | None = None
        self.noise_field: np.ndarray | None = None
        self.frame_index = 0
        self.scene_cuts = 0
        if self.options.noise_mode not in ("fresh", "frozen", "zero", "advected"):
            raise ValueError("noise_mode must be fresh, frozen, zero or advected")

    def reset(self) -> None:
        self.history = None; self.previous = None; self.noise_field = None; self.frame_index = 0

    def _noise_channels(self, geometry: NetworkGeometry, motion: np.ndarray | None, height: int, width: int) -> np.ndarray | None:
        """Replacement for feature channels 0-2 at the network extent, or None to keep the per-frame noise."""
        mode = self.options.noise_mode
        if mode == "fresh":
            return None
        if mode == "zero":
            return np.zeros((geometry.network_height, geometry.network_width, 3), dtype=np.float32)
        if mode == "frozen":
            return deterministic_noise(geometry.network_height, geometry.network_width, 0)
        if self.noise_field is None:
            self.noise_field = deterministic_noise(height, width, 0)
        elif motion is not None:
            yy, xx = np.indices((height, width))
            u = (xx.astype(np.float32) + np.float32(0.5)) / np.float32(width) + motion[..., 0]
            v = (yy.astype(np.float32) + np.float32(0.5)) / np.float32(height) + motion[..., 1]
            warped = sample_history(self.noise_field, u, v)
            fresh = deterministic_noise(height, width, self.frame_index)
            outside = (u < 0) | (u > 1) | (v < 0) | (v > 1)
            warped = np.where(outside[..., None], fresh, warped)
            ref_mean = self.noise_field.reshape(-1, 3).mean(0); ref_std = self.noise_field.reshape(-1, 3).std(0) + np.float32(1e-6)
            mean = warped.reshape(-1, 3).mean(0); std = warped.reshape(-1, 3).std(0) + np.float32(1e-6)
            self.noise_field = ((warped - mean) / std * ref_std + ref_mean).astype(np.float32)   # bilinear taps shrink the variance
        rows, columns = geometry.source_rows(), geometry.source_columns()
        return self.noise_field[rows[:, None], columns[None, :], :]

    def _controls(self) -> dict[str, float]:
        controls = dict(PROFILES[self.options.profile])
        for key in ("normalized_style", "local_tone_strength", "local_structure_strength"):
            value = getattr(self.options, key)
            if value is not None:
                controls[key] = value
        return controls

    def process(self, frame: np.ndarray, *, motion: np.ndarray | None = None, control_mask: np.ndarray | None = None, skin_mask: np.ndarray | None = None) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be (height, width, 3)")
        if self.previous is not None and (self.previous.shape != frame.shape or self._is_scene_cut(frame)):
            self.reset(); self.scene_cuts += 1
        height, width = frame.shape[:2]
        geometry = NetworkGeometry.vendor_aligned(width, height)
        controls = self._controls()
        if self.history is None:
            network = make_features(frame, frame_index=self.frame_index, geometry=geometry, control_mask=control_mask, skin_mask=skin_mask, **controls)
            noise = self._noise_channels(geometry, None, height, width)
            if noise is not None:
                network[..., 0:3] = noise
            head = geometry.crop(self.pipeline.run_features(network))
            output = compose_head(head, frame, control_mask=control_mask, intensity=self.options.intensity)
        else:
            if motion is None:
                motion = self.motion(frame, self.previous)
            features = make_temporal_features(frame, self.history, motion, frame_index=self.frame_index, control_mask=control_mask, skin_mask=skin_mask, **controls)
            network = extend_features(features, geometry, self.frame_index)
            noise = self._noise_channels(geometry, motion, height, width)
            if noise is not None:
                network[..., 0:3] = noise
            head = geometry.crop(self.pipeline.run_features(network))
            output = compose_temporal(head, frame, features, blend_scale=self.options.blend_scale, control_mask=control_mask, intensity=self.options.intensity)
            if self.options.hp_history > 0:
                output = self._blend_highpass_history(output, frame, motion, height, width)
        self.history = output; self.previous = frame; self.frame_index += 1
        return compose_detail(
            frame, output, detail_strength=self.options.detail_strength, colour_strength=self.options.colour_strength, radius=self.options.detail_radius
        )

    def _blend_highpass_history(self, output: np.ndarray, frame: np.ndarray, motion: np.ndarray, height: int, width: int) -> np.ndarray:
        """Band-split temporal blend: keep the vendor blend on the low-pass, pull the high-pass toward the
        reprojected previous OUTPUT by ``hp_history`` where the reprojected previous INPUT still matches
        the current input (photometric gate), so invented detail stops re-rolling frame to frame."""
        from .composition import blur, gaussian_kernel

        yy, xx = np.indices((height, width))
        u = (xx.astype(np.float32) + np.float32(0.5)) / np.float32(width) + motion[..., 0]
        v = (yy.astype(np.float32) + np.float32(0.5)) / np.float32(height) + motion[..., 1]
        prev_out = sample_history(self.history, u, v)
        prev_in = sample_history(self.previous, u, v)
        kernel = gaussian_kernel(self.options.hp_history_sigma)
        low = lambda img: np.stack([blur(blur(img[..., c], kernel).T, kernel).T for c in range(3)], axis=-1)
        error = np.abs(frame - prev_in).mean(-1, keepdims=True)                 # 0..1
        gate = np.clip(1.0 - error / np.float32(0.06), 0.0, 1.0)                # full trust under ~1.5/255, none above ~15/255
        alpha = np.float32(self.options.hp_history) * gate
        hp_out = output - low(output); hp_prev = prev_out - low(prev_out)
        return np.clip(output + alpha * (hp_prev - hp_out), 0, 1).astype(np.float32)

    def _is_scene_cut(self, frame: np.ndarray) -> bool:
        if self.options.scene_cut_threshold <= 0:
            return False
        luma = lambda f: f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722
        return float(np.abs(luma(frame) - luma(self.previous)).mean()) > self.options.scene_cut_threshold
