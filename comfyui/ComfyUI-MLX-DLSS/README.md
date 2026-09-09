# ComfyUI-MLX-DLSS

The DLSS neural renderer and frame generator as ComfyUI nodes (category `MLX-DLSS`).

> **Not magic.** The Neural Re-Detailer (tagline: *DLSSDetailer*) upscale/re-detail is a minor detailer: it does not invent objects or change
> the subject, it adds a small, measured amount of crispness. The fine texture is *mostly synthesised* (only
> ~30% aligns with the true detail), so enhancement, not reconstruction — a light finishing pass, most useful
> on soft/upscaled video.

## Install

```sh
ln -s /path/to/mlx-dlss/comfyui/ComfyUI-MLX-DLSS ComfyUI/custom_nodes/ComfyUI-MLX-DLSS
ComfyUI/venv/bin/pip install -e '/path/to/mlx-dlss/python[mask,video]'
```

Weights: `mlxdlss-weights all nvngx_dlssnr.dll weights/` and `mlxdlss-weights extract-fg …`
(README at the repository root). The loader nodes take a path; relative paths resolve under
`$MLXDLSS_WEIGHTS` (default `~/mlx-dlss/weights`). Restart ComfyUI after installing.

## Nodes

| node | in | out | notes |
|---|---|---|---|
| MLX-DLSS Load Neural Renderer | weights, device, precision | RENDERER | cached per (path, device, precision) |
| MLX-DLSS Neural Rendering | RENDERER, IMAGE, profile, processing_scale, detail/colour/radius/intensity, auto_mask, mask_floor, mask_feather, noise_frame_index, optional MASK | IMAGE, MASK | defaults = the Krea2 still recipe (scale 2, colour 0.5, skin auto-mask); the noise index advances per frame of a batch |
| MLX-DLSS Neural Rendering VIDEO (Metal, temporal) | IMAGE batch, model_package (.dlssmodel), binary, temporal, scene_cut, profile, scale, detail/colour/radius/intensity, precision | IMAGE, INT scene_cuts | macOS only, needs the built `mlxdlss` binary + `mlx.metallib`; ~17x the PyTorch node on video, same output within 1/255; no skin auto-mask on this path |
| MLX-DLSS Image Upscale (resample + neural re-detail) | RENDERER, IMAGE, scale_factor, method (lanczos / bicubic / upscale_model), detail/colour/intensity, auto_mask, optional UPSCALE_MODEL | IMAGE | pixels from the resampler or any spandrel model (ComfyUI's Load Upscale Model), detail from the renderer at processing scale 1; defaults = the fine-tune recipe, so load the fine-tuned weights. DLSS Super Resolution itself was measured and left out: it loses to Lanczos on video without engine motion vectors (docs/super-resolution.md) |
| MLX-DLSS Video Upscale (resample + neural re-detail, temporal) | same + hp_history, scene_cut | IMAGE, INT scene_cuts | the image path per frame through the temporal session: optical-flow history, gated high-pass history blend (0.5 default) against shimmer, scene-cut reset; needs `mlxdlss[video]` |
| MLX-DLSS Load Frame Generator | weights, device, precision | FRAMEGEN | |
| MLX-DLSS Frame Generation | FRAMEGEN, IMAGE batch, factor, scene_cut, batch | IMAGE, INT scene_cuts | `factor - 1` frames between each pair; pairs whose luma change exceeds `scene_cut` are held as a hard cut. Set the downstream frame rate to fps × factor for smooth motion, or keep it for slow motion |

`example_workflows/`: `mlxdlss_video_upscale.json` (VHS_LoadVideo → Video Upscale 2x + re-detail with the v2 fine-tune → VHS_VideoCombine — the friendly one-node video path) — the video examples need the third-party VideoHelperSuite (VHS) pack; the shipped graph forces 24 fps to stay A/V-synced, raise `frame_rate` on both VHS nodes to match a faster source, `mlxdlss_image_upscale_1.5x.json` (LoadImage → Image Upscale 1.5x with the v2 fine-tune → SaveImage), `mlxdlss_still_krea2_recipe.json` (LoadImage → renderer → SaveImage +
mask preview) and `mlxdlss_framegen_x2_video.json` (VHS_LoadVideo → frame gen ×2 →
VHS_VideoCombine at 48 fps, audio passed through). The video one needs
ComfyUI-VideoHelperSuite.

Verified: the renderer node is byte-identical to `mlxdlss-torch run` on the same still; the
frame-gen node reports the same cut and holds the same frame as `mlxdlss-video framegen`.
