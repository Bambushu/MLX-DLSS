# Rival panel on "how to get the fine-tuned weights as good as possible" — 2026-09-06

Models (independent, blind): MiniMax M3, Kimi K3, DeepSeek V4 Pro. Brief: finetune-panel-brief.md.
Full answers are in the session transcript; this is the merge.

## Consensus (3/3)
1. **Fix the degradation before touching losses.** Training inputs (0.60 hp retained) are sharper
   than deployment (RealSR 0.51, H3 likely lower). All three: replace most of the JPEG chain with a
   real **x264 round trip at CRF 30-40 / yuv420p 4:2:0**, add sensor noise, widen blur/downscale,
   keep a JPEG branch and a "clean / do-nothing" branch (~10-20%) so the net learns to leave sharp
   input alone. Kimi: calibrate against a corpus of REAL H3 frames' hp-ratio histogram, not RealSR.
   Kimi: compute the skin mask on the DEGRADED crop with inference-identical settings (train/serve skew).
2. **Synthetic-flow temporal training is sound** (all three): warp the SHARP target with an
   analytic flow (affine + low-frequency deformation, ± a foreground blob with its own affine for
   exact occlusion masks), degrade both frames independently, warp output A onto B, penalize
   high-pass-only inconsistency (Charbonnier/L1 + SSIM) with weight ~0.05-0.5, masked at
   occlusions/borders. No learned flow estimator needed (MiniMax, Kimi). Expected flicker 1.1-1.3 → ~1.05.
3. **GAN: yes but last, small, band-limited.** PatchGAN with spectral norm, fed the HIGH-PASS band
   only (never global tone), weight 0.005-0.05, warm-up then ramp, R1/EMA. Kimi's critical catch:
   outputs live on an E4M3×0.25 lattice — **round the real high-pass to the same grid before D**
   or D just detects the quantization. Expected hp 4.6 → 5.5-6.0. Sequence after the cheaper wins.
4. **Replace the single-scale 7x7 energy term** with multi-scale Laplacian matching (MiniMax:
   3 bands + local variance; Kimi: patch-wise sorted-coefficient Wasserstein = variance + kurtosis)
   plus explicit anti-halo hinges near strong edges and an anti-mottle hinge on flat regions.

## Unique, worth doing
- **Kimi #2 (inference-only, hours): noise-channel coherence.** Channels 0-2 are fresh iid noise
  per frame → shimmer exactly where detail is invented. Ablate zero / frozen / flow-advected noise;
  then split the history blend by band (more history on high-pass). Cheapest large flicker win.
- **Kimi: "semantic dropout" degrades** — extra blur on lashes/brows/lips/hairline in 35% of crops
  so the net must invent rather than amplify (why hallucination is "cautious").
- **DeepSeek: hallucination budget** — weight the texture loss up on skin/hair/fabric masks, down
  near Canny edges of the target; feather masks.
- **DeepSeek: inference post-pass** — motion-compensated blend gated by a detail-agreement map.
- **MiniMax: clean-skin targets (30%)** — median-filter skin on some targets so the net doesn't
  over-pore AI-cleaned skin; add real phone/video frames (DIV2K phone subset, Vimeo-90K).

## Disagreements
- Perceptual net: DeepSeek says swap DINOv2 for band-split LPIPS-alex; MiniMax/Kimi keep DINO
  (Kimi: keep at 1.0; MiniMax: drop to 0.5 once GAN is on). → keep DINO, test LPIPS as an ablation.
- GAN weight: 0.005 (MiniMax) vs 0.01 (DeepSeek) vs 0.05-0.1 (Kimi). → start 0.005, ramp.
- Target ceiling: Kimi argues 6.5 is a still-photo statistic; for 24 fps video aim 5.5-6.0.

## Plan (order = consensus)
A. Degradation v2 (x264 CRF 30-40 4:2:0, noise, clean branch, semantic dropout, mask on degraded
   crop, calibrate to real H3 frame histogram) → fine-tune from crisp 3k steps. Gate: RealSR ≥ 29.3,
   H3 hp up.
B. Inference: noise-channel ablation (zero/frozen/advected) + band-split history blend. Gate:
   moving-pixel flicker, detail drop < 5%.
C. Multi-scale Laplacian + anti-halo/anti-mottle hinges replacing the 7x7 energy → 3k steps.
D. Synthetic-flow temporal fine-tune, 2.5k steps, hp-only consistency λ 0.3.
E. Band-limited PatchGAN polish, 2k steps, λ 0.005→, E4M3-rounded reals. Only if C lands < 5.8.
