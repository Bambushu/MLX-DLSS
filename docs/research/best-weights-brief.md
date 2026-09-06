# Brief: last stretch to the best possible fine-tune of the DLSS 5 neural renderer (2026-09-06)

Model: NVIDIA DLSS 5 neural renderer recovered into PyTorch (71-block windowed transformer, 146M
params, FP8/E4M3 rounding at 1476 sites, fp16 gate; output = input + 0.25·highpass(head)). Trained
as-is with a straight-through estimator; 508 of 649 tensors get gradients. Local Apple M5, 51 GB
unified memory, 2.6 s/step at crop 256 batch 2. No pod.

Task: make a SOFT source (AI video such as MiniMax H3, phone footage, x264 at CRF 22-36) look like a
sharp real photograph on skin (pores, freckles, brows) without halos, mottle or shimmer, for video
(per-frame model + a photometrically gated high-pass history blend at inference).

Data: self-supervised soft→sharp pairs. Sharp = FFHQ-1024 filtered by sharpness (now all 70 shards,
~28k faces), LSDIR (40 shards), Flickr2K, DIV8K at ≤3072 px. Degradation v2: 20% clean / 10% phone
noise / 20% JPEG q35-75 / 50% x264 CRF 22-36 (30% double encode), downscale 0.55-1.0, blur 0-0.7,
calibrated to real H3 frames (skin high-pass 2.28 vs FFHQ sharp 3.05). Skin mask from a SegFormer
face parser, computed on the degraded crop, fed to the network's skin channels and used to weight
the loss. RealSR V3 for calibration/eval.

Loss that works: low-pass L1 + high-pass L1 (0.5) + local high-pass energy (2.0) + DINOv2-base
patch-token cosine (1.0). Being added: multi-scale Laplacian band L1 + anti-halo/anti-mottle
hinges (per-band contrast matching diverged), FP8 envelope barrier on activations (activation
blow-up above E4M3 max 448 was the divergence mode), synthetic-flow two-frame temporal consistency
on the high-pass, band-limited spectral-norm PatchGAN with hinge loss + feature matching on
E4M3-quantized reals (weight 0.005, warm-up 200, ramp 500). AdamW lr 3e-6, cosine, grad clip 1.0.
Planned: EMA 0.999, soups of checkpoints, lr sweep 1e-6..1e-5, late-blocks-only, MUSIQ/CLIP-IQA/
TOPIQ + LPIPS/DISTS ranking, then a blind A/B.

Measured so far: fine-tune v2 (honest) and v1-crisp (punchy) both add real pore detail on soft H3
faces; stock weights do nothing on soft input. Noise channels are ignored by the network. Scale 2
makes fine-tunes softer. Flicker on static regions handled at inference by the history blend.

Question: what are we still missing that would measurably improve realism or reduce artifacts,
within local compute (a few hours per run)? Rank by expected gain per hour. Be concrete: exact
loss/term/schedule/data/inference change, and how to verify it.
