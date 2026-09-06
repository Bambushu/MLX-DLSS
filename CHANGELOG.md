# Changelog

## 0.2.0 (fork, 2026-09) — Bambushu/MLX-DLSS

- Fine-tuning: `scripts/finetune.py` trains the recovered renderer (STE on E4M3 rounding, FP8
  envelope barrier, real-photo soft→sharp pairs, DINOv2 perceptual, optional Laplacian / temporal /
  adversarial terms); `scripts/fetch_datasets.py`; fine-tuned weight sets v1, v1-crisp, v2.
- Renderer: per-pixel skin auto-mask (`--auto-mask skin`, face parsing into channels 13/14),
  control masks resampled at processing scale, `--hp-history` gated high-pass history blend,
  `--noise-mode` for the temporal session.
- Frame generation: `--scene-cut` gate (hold a hard cut instead of blending across it), reported
  cut indices.
- ComfyUI: `comfyui/ComfyUI-MLX-DLSS` (renderer, Metal video renderer, frame generation, loaders)
  with example workflows.
- Metal: builds on Command Line Tools via the prebuilt `mlx.metallib` (`MLXDLSS_METALLIB`).
- Validation: `scripts/validate.py` (frame-gen PSNR vs RIFE 4.7 / 4.26 / minterpolate, renderer
  sharpness, static-vs-moving flicker); results in `ab/RESULTS.md`.
