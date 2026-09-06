# Question: how do we get these fine-tuned weights as good as possible? Rank the next steps.

## Setup (all real, measured; ask nothing, answer from this)
- Model: NVIDIA DLSS 5 "neural renderer" recovered into PyTorch (windowed transformer, 71 blocks,
  146M params, 16-channel per-pixel input: noise ×3, constant, colour ×2 copies, style idx, tone,
  structure, skin-mask ch13, auto-mask ch14). Output head: 4 ch; image = input + 0.25 * head[:3]
  (residual). Everything is E4M3-rounded inside; we train it with a straight-through estimator
  on every rounding, full float32, all 508 weight tensors trainable (attention bias/scale frozen).
- Goal: a detail/"photoreal" pass for SOFT AI-generated video frames (MiniMax H3 output, 800x1440
  h264, ~0.06 bpp) and soft phone footage. Runs per frame, then a temporal history blend with
  optical-flow reprojection (inference only, not trained). Must stay temporally stable.
- Hardware: ONE Apple M5 Pro, 48 GB unified memory, PyTorch MPS. No CUDA. 256 crop batch 2 fwd+bwd
  = 2.0 s. No pods allowed. Training budget: hours to a couple of days.
- Data: self-supervised soft→sharp pairs. Sharp target = real photos: FFHQ-1024 filtered by a
  sharpness metric (4834 kept of 12k; half of FFHQ is soft upscales) + LSDIR (1745, ≥768px).
  NO AI-generated stills allowed in training. Input = degrade(target): downscale 0.35-0.65 (Lanczos)
  → resample back (bilinear/bicubic/lanczos) → JPEG q25-55 → gaussian blur σ0.5-1.3. Calibrated on
  RealSR (real DSLR x2 pairs keep 0.51 of the sharp high-pass energy; our degrade keeps 0.60).
  Crops 256-288, skin-biased 70%; skin mask (face-parsing SegFormer) fed to ch13/14.
- Losses tried:
  * L1 + hp-L1 only: collapses to identity in 20 steps (regression to mean).
  * low-pass L1 + 0.5 hp-L1 + 4.0 local high-pass ENERGY match (|hp| avg-pooled 7x7): texture
    amount matches target but over-sharpens real photos into halos; PSNR on RealSR real pair
    drops below the soft input (27.2 vs 29.3). Plateaus then degrades.
  * + DINOv2-base patch-token (1−cos) perceptual 1.0, energy 2.0 (run3): PASSES. Held-out PSNR
    31.3 (soft input 31.15, stock net 28.4), high-pass 4.4-5.2 vs target 6.5 vs soft 2.5.
    RealSR real pair PSNR 29.39 (> LR 29.30). No halos. This is shipped "v1".
  * run4 = same + cosine decay, 6000 steps, crop 288: converged by 4000; slightly crisper
    edges; RealSR 28.89 (a bit over-sharp on real photo). Shipped as "crisp", preferred on video.
  * run5 ablation DINO 2.0 / energy 1.0: best PSNR 31.5 but half the texture (3.3); too soft.
- Where it stands: on H3 video frames the fine-tune clearly adds lash/brow/freckle/pore/fabric
  detail at 100% with no halos, keeps colour (stock net desaturates). Detail energy reaches ~4.6
  of the 6.5 target — the remaining gap is cautious hallucination. Temporal: flicker ratio vs
  source 1.1-1.3 in MOVING pixels only (static pixels ≤ stock), i.e. added detail moving with the
  subject; visible as slight "shimmer" on hair/fabric in motion at 24 fps.
- Constraints: no video soft/sharp pairs exist. No CUDA. Model architecture fixed (recovered
  vendor graph); we can add losses, data, schedules, small trainable adapters, but not change the
  network. Inference must remain per-frame (+ the existing history blend).

## Deliver
1. Ranked list of the 3-5 most valuable next steps to (a) close the remaining detail gap without
   halos and (b) reduce moving-pixel flicker. For each: concrete recipe (loss form, weights,
   data construction, schedule), expected gain, main risk, and how to measure it.
2. Specifically judge: adversarial (GAN) loss on this residual model on MPS — worth it? which
   discriminator, which weight, how to keep it stable in a few thousand steps?
3. Specifically judge: temporal-consistency training WITHOUT video pairs (warp a still with a
   synthetic flow field, run both frames, penalize inconsistency after warping the output) —
   sound? what flow model/warp, what loss, what pitfalls?
4. Anything we are doing wrong in the degradation model or data that you would fix first.
Be concrete and opinionated. No generic advice.
