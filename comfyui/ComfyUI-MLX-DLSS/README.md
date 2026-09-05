# ComfyUI-MLX-DLSS

The DLSS neural renderer and frame generator as ComfyUI nodes (category `MLX-DLSS`).

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
| MLX-DLSS Load Frame Generator | weights, device, precision | FRAMEGEN | |
| MLX-DLSS Frame Generation | FRAMEGEN, IMAGE batch, factor, scene_cut, batch | IMAGE, INT scene_cuts | `factor - 1` frames between each pair; pairs whose luma change exceeds `scene_cut` are held as a hard cut. Set the downstream frame rate to fps × factor for smooth motion, or keep it for slow motion |

`example_workflows/`: `mlxdlss_still_krea2_recipe.json` (LoadImage → renderer → SaveImage +
mask preview) and `mlxdlss_framegen_x2_video.json` (VHS_LoadVideo → frame gen ×2 →
VHS_VideoCombine at 48 fps, audio passed through). The video one needs
ComfyUI-VideoHelperSuite.

Verified: the renderer node is byte-identical to `mlxdlss-torch run` on the same still; the
frame-gen node reports the same cut and holds the same frame as `mlxdlss-video framegen`.
