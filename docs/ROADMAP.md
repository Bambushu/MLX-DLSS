# Fork roadmap (Bambushu/MLX-DLSS)

Evidence for every item is in `ab/RESULTS.md`. Order = build order. Each item lists the gate
that decides whether it ships.

## 1. Scene-cut gate for frame generation  (~1 day) — DONE
- Problem: interpolating across a cut yields a ghost blend (ab/strip_cut.png).
- Design: per pair, luma-histogram distance + mean abs diff on the 2x2-box candidates the graph
  already computes; above threshold, emit a duplicate of frame A (fps mode) or hold (slowmo)
  instead of the generated frame. Reuse the renderer's existing `--scene-cut` luma-jump code.
- CLI: `--scene-cut [thresh]` on `mlxdlss-video framegen`; log the cut indices.
- Gate: the faces|yoga concat produces no blended seam; PSNR on the two clean clips unchanged.

## 2. Skin / face auto-mask for the renderer  (1-2 days) — DONE (per-pixel channels 13/14, not the control mask; see RESULTS)
- Problem: channels 13-14 (vendor's skin/auto mask) are zero in the port, so detail is applied
  uniformly; on soft sources that is mostly tone shift, on sharp sources it also textures hair
  and background.
- Design: optional `--auto-mask face|skin|none` in the Python pipeline; a face parser (BiSeNet
  face-parsing, CPU/MPS, already used elsewhere on this machine) produces a skin mask, blurred,
  fed as the green/blue control-mask channels with structure weight on skin only.
- Gate: Krea2 still, standard profile: pores on skin, hair and background byte-identical to
  input outside the mask (md5 of masked regions).

## 3. ComfyUI node pack  (1-2 days) — DONE (comfyui/ComfyUI-MLX-DLSS, symlinked into ~/ComfyUI-h3)
- Two nodes: `MLXDLSS Frame Generation` (IMAGE batch in, multiplier, scene-cut) and
  `MLXDLSS Neural Rendering` (IMAGE in, profile, scale, detail, colour, intensity, optional MASK).
- Wraps the existing `FrameGenerator` / `NeuralRenderingPipeline` Python API. Device auto
  (MPS local, CUDA on pods). Ships `example_workflows/` so nobody hand-builds a graph.
- Gate: node output byte-equal to the CLI output on the same frame.

## 4. Validation on real + AI footage  (1 day) — DONE (scripts/validate.py; RIFE 4.26 is the quality pick, DLSS the speed pick)
- Extend `ab/` into a repeatable script: withheld-frame PSNR for FG (vs RIFE 4.7 and 4.26),
  no-reference sharpness + flicker ratio for the renderer, on H3 / Krea2 / real phone footage.
- Gate: numbers published in RESULTS.md; decides whether 5 and 6 are worth doing.

## 5. Real motion vectors + depth into frame generation  (1-2 weeks, research)
- The ported graph runs the vendor's MV dilation / depth splat / occlusion path but with zero
  MVs and flat depth, so it collapses to a 2-frame blend. Feed DIS optical flow as MVs and
  Depth-Anything-v2 as depth; occlusion weights and warped candidates become real.
- Risk: no vendor capture exists with real MVs, so correctness is judged only by PSNR on withheld
  frames. Payoff is bounded: item 4 shows we are already at RIFE parity, so this ships only if
  it beats RIFE 4.26 by >0.5 dB on motion clips.

## 6. Photoreal fine-tune of the renderer  (2-4 weeks, research)
- Weights are extracted to safetensors; the PyTorch reference graph is inference-only (no
  training loop, fused-attention approximations, FP8 rounding baked in). Needs: a trainable
  reimplementation of the window-attention blocks, a paired dataset (soft→sharp real video
  frames, e.g. h264-crushed vs original), a CUDA pod.
- Payoff: fixes the "does nothing on soft sources" result and the game prior. Highest value,
  highest cost. Only after 2 and 4.

## 7. Speed  (2-3 days) — DONE (Metal backend built; 17x on temporal video, batching on MPS gives nothing)
- Build the Swift/Metal backend (`swift build`, needs ninja for the metallib script) and
  benchmark 1080p video end to end; fused Metal is the author's 2x path.
- Python: batch frames through the renderer (currently 1/frame), fp16 on MPS.
- Gate: 15 s 800x1440 clip through the renderer in <5 min locally (now ~27 min).

## Deferred / rejected
- Redoing the reverse engineering: no. The recovered graphs are the asset.
- Super resolution: the author's measurement stands (loses to Lanczos); we upscale with LTX-2.5.
