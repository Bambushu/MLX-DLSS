# Best-Weights Panel — three-model review

Brief: `docs/research/best-weights-brief.md`. Run 2026-09-06 via the rival companion (`--auto 1/2/3`).

Models:
- **A — minimax/minimax-m3**
- **B — moonshotai/kimi-k3**
- **C — deepseek/deepseek-v4-pro**

---

## Consolidated ranked list (agreed by 2+ models)

1. **DROP the full LR sweep 1e-6..1e-5** — all three. Replace with a 100-step LR finder on a 1k subset (A), or param-group LRs in one run: head/detail 3x, gates 2x, rest 3e-6 (B), or two 500-step probes at 5e-6 / 7e-6 (C). Verify: train-loss slope at matched steps.
2. **Restrict high-frequency supervision to where detail actually exists / belongs.** C: mask HF L1 + local HF energy to the top-decile high-pass magnitude of the *sharp target* inside the skin mask. A: replace the binary skin mask with a soft Gaussian falloff (sigma ~6 px) on all HF terms, keep low-pass unmasked — the binary boundary is the edge-artifact generator. B (adjacent): 70% skin-centred crops so >=60% of batch pixels are skin. Verify: HF std in flat cheek/forehead stays low; the hair/forehead seam disappears.
3. **Keep the FP8 activation-envelope barrier, but make it soft.** C: `1e-4 * mean(relu(a-448)^2)` from step 0, monitor p99 activations settling <440. A: also tanh-bound the residual before `0.25*highpass`, `0.25*tanh(head/tau)`, tau~3 — check the activation histogram has no pile-up at 448. B: warns a hard clamp zeroes gradients exactly where you need them; prefer the soft excess penalty.
4. **Keep multi-scale Laplacian band L1, and make sure a mottle band exists** — all three. A: sigma 1-2 w=1.5, 2-4 w=1.0, 4-8 w=0.5, ramp mid+high over 500 steps, replaces "local high-pass energy". B: explicitly weight a 4-16 px band. C: 4-level pyramid 1.0/0.5/0.25/0.125, keep local HF energy at the finest level only. Verify: band-wise Laplacian error; mottle-band energy ratio out/target on skin.
5. **Trim the IQA zoo** — all three. Keep one perceptual ranker (TOPIQ face-tuned, or LPIPS) for *checkpoint selection only*; drop CLIP-IQA (semantic, noisy for skin) and never use these as a training loss. B replaces them with three targeted metrics: pore-band (2-6 px) energy ratio in skin, edge-band overshoot, temporal flicker on static skin.
6. **Get a real held-out H3 eval set, and let it be the arbiter.** A: shoot 30-50 paired real frames, static faces, 3 lighting setups, locked tripod, ranking only. B: additionally feed 10-20% of *training* batches as real H3 crops with no reconstruction loss (HF energy matching to an FFHQ-sharp reference + chroma guard only) to close the x264-simulation domain gap. Verify: blind A/B on held-out real H3 clips, run once at the end.
7. **Bigger crop beats more steps.** A: 256 -> 320 (384 if memory holds), a 2 px pore is a much larger fraction of the band. B: cache the DINOv2 *target* features (they are fixed per crop; 15-25% step-time cut) and spend the saving on crop 352-384, which also gives the PatchGAN a real receptive field. Verify: halo metric at matched **wall-clock**, not matched steps.
8. **Add an anti-halo term scoped to an edge band** (A and C's alternative both aim here; C prefers to skip it). A: directional Sobel on a 3-5 px annulus (`dilate XOR erode`), magnitude L1 w=1.0 + angle `1-cos(dtheta)` w=0.5. B: one-sided log-ratio hinge `softplus(log(g_out/g_tgt) - log 1.15)` per band inside a 3 px dilated strong-edge band, w=0.05, warm-up 300 steps — one-sided + log-space + tiny weight is what the diverged version lacked. Verify: mean `relu(|HP_out|-|HP_tgt|)` in the edge band on the 10 worst clips.
9. **Train through the inference operator / add temporal consistency.** B: apply the photometric-gated history blend *inside* the loss for the last 1-2k steps (gate detached), then grid-search gate thresholds offline. C: warped high-pass L1 between frame t and t+1 under a flow-confidence mask, w=0.5 after 500 steps warm-up. Verify: per-pixel std on flow-confident static skin; expect >=30% flicker drop (C), pore-band metric must not fall >5% (B).
10. **Keep EMA; the soup is contested.** A: decay 0.9 -> 0.999 over the first 1000 steps then constant (fixed 0.999 has a 1000-step window, so the tail of a 5k run barely registers), then soup the last 3. C: EMA 0.999 + soup the last 5-10 checkpoints. B: drop souping, EMA already is a tail soup.

**Contested — no consensus:**
- **Late-blocks-only fine-tune.** A and B drop it (A: "within noise at this scale, do it as a 30-min ablation, don't block on it"). C ranks it #2 and argues the detail style lives in the late blocks and freezing early ones buys steps/hour plus regularisation.
- **GAN.** C keeps PatchGAN + VGG feature matching at 0.005. B drops feature matching, keeps only the hinge at 0.005. A keeps it but says **initialise the discriminator from Real-ESRGAN/GFPGAN** — a from-scratch spectral-norm PatchGAN inside 5k steps is the usual reason GAN fine-tunes do nothing; D loss should land in [0.3, 0.6] at step 1, not 0.69.

---

## Unique to one model

**A — minimax/minimax-m3**
- Inference-only, zero training cost: TTA x4 (flip + rot90) plus multi-scale (1.0 + 0.75, Hann-overlap blend); tanh the *ensembled* residual, not each member. Called the best remaining ROI move.
- Lab chromaticity L1 on the soft skin mask, w=0.5 — nothing currently protects hue. Verify with dE2000 per skin region; FFHQ under-trains darker tones, so sample-weight by FST if the gap shows.
- Anti-mottle LoG energy match, sigma 1.5-3 px, w=0.5, minus the sigma=4 LoG to kill DC. Verify on the 2D FFT of the skin region: the 8-16 cycle/face blob should flatten.
- Drop stochastic depth / drop-path — at 146M params over 5k steps the regularisation signal is buried in FP8 noise.
- Drop investigating the ignored noise channels in the main run: zero them at inference (one line) or run a separate 1 h ablation.

**B — moonshotai/kimi-k3**
- **Audit the 141 dead tensors and FP8 update absorption first, before any more training.** Which tensors get no gradient (if head/gate tensors are among them you are tuning around the detail generator)? And measure per-step |dW| against the E4M3 ULP per tensor — at lr 3e-6 with FP8 master weights most AdamW updates round to zero. If >30% of grad-receiving tensors have median |dW| < ULP/2, move to fp32 master weights with quantize-on-forward STE, or stochastic rounding + error feedback.
- Also zero the skin-mask *input* channel at inference and measure delta; if it behaves like the ignored noise channels, the conditioning is illusory (loss-side weighting still works).
- **Drop the low-pass L1 to target.** With `input + 0.25*highpass(head)` the model cannot move low-pass; that term pushes low-frequency corrections through a leaky high-pass filter, a textbook mottle generator. Replace with a low-pass anchor to the *input* at w=0.1.
- Move HF L1 + local HF energy to **luma only**, add a chroma high-pass excess hinge `mean(relu(|HP(C_out)|-|HP(C_tgt)|))` inside the skin mask, w=0.5, aimed at colour speckle.
- Fix the scale schedule: restrict downscale to 0.8-1.0 and re-upsample the input to native target resolution — inference is 1:1, do not teach upscaling. Drop scale-2 entirely (already measured softer).
- Free inference sweep before any A/B: clamp per-band residual gain to <=1.2x a running target statistic; sweep global residual gain 0.25 -> {0.20, 0.30, 0.35} with skin-mask modulation `x(1 + 0.3*mask)`.
- Sequencing: Run A = audit + loss surgery + sampling fix; Run B = + DINOv2 caching/crop 352 + halo hinge; Run C = + train-through-blend + real H3 batches. "If only one run happens, it's Run A."

**C — deepseek/deepseek-v4-pro**
- Keeps the anti-halo/anti-mottle hinges on the DROP list: target-detail masking + multi-scale Laplacian + temporal consistency already cover it, and the hinges add two sensitive hyperparameters (band threshold, hinge margin).
- Considers per-band contrast matching permanently discarded — its divergence is evidence the FP8 net cannot take it.
- FID on high-pass-filtered skin patches as the GAN acceptance test: >=2 FID improvement at 3k steps while edge halos do not increase.

---

<details>
<summary>Raw answer — A (minimax/minimax-m3)</summary>

Auto-selected model #1: minimax/minimax-m3


Ranked by expected gain per hour on your M5. Items 1–4 are highest priority because they address artifacts the loss currently cannot see.

---

**1. Bump crop 256 → 320 (test 384 if memory holds).** At 256 a 2 px pore is 2/256 of a cycle; the FP8 STE rounds that band hard. 320 gives ~25% more headroom in the highest band the network can write into, 384 gives ~56%. Cost: 30 min to benchmark; batch 2 at 320 should fit, batch 1 at 384 is the fallback. Verify: pores visible at native scale on the same H3 crops you used to calibrate degradation v2; check that LPIPS-DISTS on skin region drops while your high-pass L1 stays flat (you want resolution, not amplitude).

**2. Anti-halo directional Sobel on a 3–5 px annulus around the skin mask.** Compute Sobel(ŷ) and Sobel(y) inside `mask_dilated XOR mask_eroded`, penalize both magnitude mismatch (L1, weight 1.0) and angle mismatch (`1 − cos(θ_out − θ_gt)`, weight 0.5). This is what your "per-band contrast matching" was *trying* to do; the annular geometry is what makes it actually bite on halos at the eye/lip/brow border instead of just smoothing everywhere. Cost: 1 h. Verify: render a thin Sobel-energy map of the residual inside the annulus on v2 vs new; halos show as a bright ring at the boundary.

**3. Soft skin mask via 5–7 px Gaussian falloff from the distance transform.** Right now the binary boundary creates a discontinuity in the loss gradient that the network satisfies by pushing garbage into the mask edge — that is your edge artifact mode. Use `mask_soft = exp(−d²/2σ²)` clipped to [0,1], σ=6 px. Apply to all *high-frequency* terms; keep low-pass L1 unmasked. Cost: 30 min. Verify: A/B on side-lit faces; the soft mask should remove the seam where hair meets forehead.

**4. Anti-mottle LoG energy match, σ=1.5–3 px on skin mask.** Mottle lives in a specific Laplacian band your existing high-pass L1 doesn't isolate. Add `‖LoG_σ(ŷ) − LoG_σ(y)‖₁` weighted 0.5, masked, only mid-band LoG (subtract the σ=4 LoG to remove DC). Cost: 1 h. Verify: 2D FFT of skin region on H3 frames — the 8–16 cycle/face blob should flatten; if you see energy *growing* there, lower the weight.

**5. Tanh-bound the residual before `0.25·highpass` and FP8 quant.** Replace `0.25·head` with `0.25·tanh(head/τ)` with τ≈3 (so linear region is ±3σ). Prevents the E4M3-max-448 envelope barrier from being the bottleneck at every divergence. Cost: 15 min. Verify: histogram of activations at the FP8 boundary sites; pre-fix you should see a tail at 448, post-fix it should be empty.

**6. Inference-only, zero training cost: TTA × 4 (flip + rot90) and multi-scale (1.0 + 0.75 blend with Hann overlap).** You already pay the model cost per frame; paying it 4× for an 8% perceived-quality lift on the exact artifact class you're chasing (pores, which are rotation-equivariant) is the best ROI move you have left. Tanh the *ensembled* residual, not each individual one. Cost: 1 h to wire up. Verify: side-by-side on 20 H3 frames you didn't train on; pores should look less aliased, no new halos.

**7. Init the PatchGAN discriminator from Real-ESRGAN's or GFPGAN's pretrained D.** Training a spectral-norm PatchGAN from scratch inside 5000 steps of a frozen generator is the most common reason GAN fine-tunes "do nothing" or mode-collapse on skin. Reuse the conv stack; only re-init the final patch head for your 1-channel quantized-real input. Cost: 30 min. Verify: D loss should land in [0.3, 0.6] from step 1, not start at 0.69 (saturated, learning nothing).

**8. Collect 30–50 paired real H3 frames as a held-out validation set.** You don't have ground truth right now — RealSR V3 is a proxy and your IQA stack is a noisier proxy. Spend one afternoon shooting static faces under three lighting setups with a locked tripod, then use those frames *only* for ranking. Everything else (perceptual losses, ranking metrics) becomes secondary. Cost: 2–3 h. Verify: this *is* the verification for items 1–7 and for v2 vs v1-crisp.

**9. Per-band Laplacian L1 — confirm band split, don't reinvent.** σ=1–2 (high, weight 1.5), σ=2–4 (mid, weight 1.0), σ=4–8 (low, weight 0.5). The mid-band term replaces your "local high-pass energy" and is more interpretable. Ramp mid and high together over 500 steps; keep low constant. Already on your list; just nail the schedule.

**10. EMA decay schedule 0.9 → 0.999 over first 1000 steps, then constant, then 3-checkpoint soup.** With ~5000-step runs, fixed 0.999 has an effective window of 1000 steps and your last 200 steps barely register. Starting fast then slowing gives both stability and end-of-run responsiveness. Soup the last three. Cost: trivial. Verify: validation set at each EMA checkpoint.

**11. Lab chromaticity L1 on skin mask, weight 0.5.** You have no term that protects hue. `‖ab(ŷ) − ab(y)‖₁` weighted 0.5 inside the soft mask catches color drift cheaply. Cost: 1 h. Verify: ΔE2000 in skin region, A/B across skin tones (FFHQ under-trains darker tones — sample-weight your data if ΔE gaps by FST).

---

**Drops (explicit):**

- **LR sweep 1e–6..1e–5** — replace with a 100-step LR finder on a 1k-subset. Same info, ~10× cheaper.
- **Stochastic depth / drop-path** — at 146M params and 5k-step runs, the regularization signal is buried in the FP8 noise.
- **Late-blocks-only fine-tune** — likely within noise of full fine-tune at this scale; if you want to know, do it as one 30-min ablation after the main run, don't block on it.
- **MUSIQ + CLIP-IQA + TOPIQ all three** — CLIP-IQA is semantic and noisy for skin texture, MUSIQ is general. Keep TOPIQ (face-tuned variant) plus LPIPS/DISTS; use your held-out H3 set (item 8) as the actual arbiter. Blind A/B stays, but on *real* H3 frames with controlled capture, not proxy clips.
- **Investigating noise channels in the main run** — you've measured they're ignored. Either zero them at inference (1 line) or do a separate 1 h ablation training a copy without them. Don't spend the main run's compute on it.

</details>

<details>
<summary>Raw answer — B (moonshotai/kimi-k3)</summary>

Auto-selected model #2: moonshotai/kimi-k3
## Ranked by expected gain per hour

**1. Audit the 141 dead tensors and FP8 update absorption (≈1 h, no training) — do this first.**
Two silent-failure checks before any more runs:
- *Which* tensors get no gradient? If head/gate tensors are among the 141, you're tuning around the detail generator. Either route STE through them or freeze deliberately and compensate with head lr.
- Measure per-step |ΔW| against the E4M3 ULP per tensor. With lr 3e-6, if master weights are FP8, most AdamW updates are sub-ULP and round to zero — the run "works" only via the few largest-magnitude weights. If >30% of grad-receiving tensors show median |ΔW| < ULP/2, switch to fp32 master weights with quantize-on-forward (STE), or stochastic rounding + error feedback.
- Also: zero the skin-mask input at inference and measure Δ. "Noise channels are ignored" suggests input channels can be dead; if the mask channel behaves the same, your conditioning is illusory (the loss-side weighting still works).
Verify: fraction of weights changed after 500 steps; train-loss slope; pore-band metric (below) at matched step count.

**2. Loss surgery for mottle (one run).**
- **Drop low-pass L1 to target.** Your architecture is `input + 0.25·highpass(head)` — it *cannot* move low-pass. That term forces low-frequency corrections through a leaky high-pass filter, which is a textbook mottle generator. Replace with a low-pass anchor to the *input*, weight 0.1 (prevents drift, removes the impossible objective).
- Move high-pass L1 + local high-pass energy to **luma only**; add a chroma high-pass excess hinge `mean(relu(|HP(C_out)| − |HP(C_tgt)|))` inside the skin mask, weight 0.5. This targets color speckle directly.
Verify: mottle-band (8–32 px) energy ratio output/target in skin on a fixed val set; chroma HP energy; 2-model A/B.

**3. Skin-dense sampling + scale fix (data-side, ~0 cost).**
- 70% of crops centered on skin via the parser, 30% random; target ≥60% skin pixels/batch. Random 256-crops on 1024-px images are mostly not-skin — this is a ~3× multiplier on effective skin supervision.
- Restrict downscale to 0.8–1.0 and **re-upsample the input to native target resolution** (inference is 1:1; don't teach upscaling). Drop scale-2 entirely — you already measured it goes softer.
Verify: logged skin-pixel fraction; radial high-pass spectrum of outputs on H3 val vs the FFHQ-sharp reference.

**4. Cache DINOv2 target features; spend the savings on context.**
Targets are fixed per crop → precompute patch tokens once, halving DINOv2 forwards (expect 15–25% step-time cut). Reinvest at crop 352–384 at the same s/step. Halos are context-limited; bigger crops also give PatchGAN a real receptive field.
Verify: s/step before/after; halo metric at matched wall-clock, not matched steps.

**5. Anti-halo that won't diverge + a free inference clamp.**
- Replace per-band contrast matching with a **one-sided log-ratio hinge**: `softplus(log(g_out/g_tgt) − log 1.15)` per Laplacian band, computed only inside a 3-px dilated strong-edge band (edges from the target low-pass), weight 0.05, warm-up 300 steps. One-sided + log-space + tiny weight is what your diverged version was missing.
- Zero-training inference wins: clamp per-band residual gain to ≤1.2× a running target statistic, and sweep the global residual gain 0.25 → {0.20, 0.30, 0.35} with skin-mask modulation (×(1 + 0.3·mask)). Do this before any A/B.
Verify: edge-band overshoot ratio (mean relu(|HP_out|−|HP_tgt|) in the edge band) on the 10 worst clips.

**6. Train through the inference operator (last 1–2k steps of a run).**
Apply the photometric-gated history blend *inside the loss* on your synthetic-flow pairs (gate detached, loss on the blended output). Then grid-search the gate thresholds on synthetic clips — 30 min, no training. This closes the train/inference mismatch that the per-frame model + post-hoc blend otherwise guarantees.
Verify: temporal flicker (per-pixel std on flow-confident static skin) vs per-frame baseline; pore-band metric must not drop >5%.

**7. Real H3 unpaired batches (domain gap).**
x264 simulation is calibrated in energy (2.28 vs 3.05) but not in *structure* — AI video has waxy, temporally unstable texture that CRF sweeps don't produce. Add 10–20% of batches as real H3 crops with **no reconstruction loss**: luma high-pass energy matching to the FFHQ reference + chroma guard + temporal consistency through the blend. No GAN on these initially.
Verify: blind A/B on held-out real H3 clips; watch for waxy drift.

**8. Param-group lr instead of a sweep.**
Head/detail tensors at 3× (9e-6), gates at 2×, rest at 3e-6 — one run. Replaces the entire sweep.

## DROP (explicitly, from your planned/in-flight list)

- **lr sweep 1e-6..1e-5** — drop; replaced by #8. Each point costs a full run and the FP8 barrier caps the useful range anyway.
- **Checkpoint soups** — drop; EMA 0.999 is a tail-soup you get for free. Keep EMA, skip souping.
- **Late-blocks-only** — drop as planned; fund context via DINOv2 caching instead (#4). Revisit only if the audit shows early blocks eat >50% of step time.
- **The IQA zoo (MUSIQ/CLIP-IQA/TOPIQ + LPIPS/DISTS)** — drop all but one (keep LPIPS or TOPIQ). Replace with three targeted metrics: pore-band (2–6 px) energy ratio in skin, edge-band overshoot, temporal flicker on static skin. Cheaper, and they actually track your failure modes.
- **GAN feature matching** — drop; keep the hinge at 0.005. Feature matching on E4M3-quantized reals is redundant with high-pass energy + chroma guard and costs step time.
- **Low-pass L1 to target** — drop (see #2); prime suspect for mottle.
- **Blind A/B** — keep, but run it *once* at the end on a fixed eval set (~30 H3 + 10 phone clips). Don't spend hours A/B-ing mid-sweep.

Keep: EMA, multi-scale Laplacian bands (make sure a 4–16 px mottle band exists and is weighted), the FP8 barrier — but verify gradient flow past it (a hard clamp zeroes gradients exactly where you most need them; prefer a soft penalty on the excess over 448).

## Sequencing

Run A: #1 fixes + #2 + #3. Run B: + #4 (crop 352) + #5 hinge. Run C: + #6 tail + #7. Then the free inference sweep (#5) and a single blind A/B. If only one run happens, it's Run A — the audit and the low-pass/mottle fix dominate everything else.

</details>

<details>
<summary>Raw answer — C (deepseek/deepseek-v4-pro)</summary>

Auto-selected model #3: deepseek/deepseek-v4-pro
### Ranked suggestions by expected gain per hour (descending)

All changes are designed to be tested in a few local runs of a few hours each, leveraging the current 51 GB M5 setup with 2.6 s/step (crop 256, batch 2). The ranking prioritises improvements that directly attack the most common video upscaling artifacts (mottle, halos, shimmer) while avoiding needless hyperparameter churn.

---

#### 1. **Target‑detail‑guided loss masking** (immediate, no extra compute)
**Change**  
Multiply the existing `high‑pass L1` and `local high‑pass energy` losses (before the global averaging) by a binary mask:
- Compute the high‑pass magnitude of the **sharp target** (same filter as used for the loss).
- Threshold at the 90th percentile of that magnitude inside the skin mask (to focus on real pores, brow hairs, etc.).
- During training, only back‑propagate these two losses where the mask is 1.
- Keep the `low‑pass L1` and `DINOv2` losses unmasked so colour and coarse structure are still learned everywhere.

**Why**  
Soft inputs often contain flat skin regions where the target has no micro‑detail. Without masking, the network is forced to invent texture that never existed, producing mottle or halos. By restricting high‑frequency supervision to genuinely detailed areas, you prevent hallucinated grain while preserving real detail where it matters.

**Verification**  
Run a blind side‑by‑side on a few test faces (soft H3 video frames). Measure the standard deviation of the high‑pass in skin‑masked cheeks and foreheads: it should stay low only where the target is smooth, and rise appropriately near pores/brows. Any diffuse “sandiness” should disappear.

---

#### 2. **Late‑blocks‑only fine‑tuning** (~30 % faster iteration)
**Change**  
Freeze the first 40 (of 71) windowed‑transformer blocks and only train the last 31 blocks + the output head. This reduces trainable parameters from ~146 M to ≈85 M.
– Keep the same AdamW lr of 3e‑6, cosine decay, and grad clip 1.0 initially; optionally raise lr slightly for the unfrozen layers (e.g., 5e‑6) after verifying stability.
– All other planned losses and data remain identical.

**Why**  
The recovered DLSS 5 model already captures universal low‑level features in its early blocks. The detail‑generating “style” is predominantly in the later layers. Freezing early blocks not only speeds up each step (→ more effective steps per hour) but also regularises against overwriting previously learned robust representations, reducing risk of over‑sharpening or colour shifts.

**Verification**  
After 2 k steps, compare PSNR and LPIPS on a fixed validation set (degraded‑sharp pairs) against a full‑model fine‑tune run of equal wall‑clock time. Visual check: early‑block‑frozen outputs should show equally crisp pores but fewer artifacts on glasses/hair boundaries.

---

#### 3. **Synthetic‑flow temporal consistency on high‑pass** (high gain for video shimmer)
**Change** (planned – keep and prioritise)  
During training, for each 2‑frame pair (consecutive crops from the same video source), compute optical flow (e.g., a pre‑computed lightweight off‑the‑shelf flow or a frozen small flow net) and warp the high‑pass output of frame t to time t+1. Add an L1 loss between the warped high‑pass and the actual high‑pass from frame t+1, weighted by a flow‑confidence mask. Weight = 0.5 (relative to other losses). Apply only after 500 steps of warm‑up to let the base detail settle.

**Why**  
Even with the inference‑time photometric history blend, shimmer can appear under fast motion when the blend fails. Training with a direct temporal smoothness constraint on the network’s own texture output forces it to produce inherently stable detail, reducing reliance on post‑processing.

**Verification**  
Render a short video of a talking head with small panning motion. Calculate the temporal flicker metric (mean absolute frame‑to‑frame difference in high‑pass within a stable skin region) before and after. The measure should drop by at least 30 % compared to the temporally‑unaware baseline.

---

#### 4. **FP8 activation envelope barrier** (stability prerequisite)
**Change** (planned – keep)  
Add a penalty:  
`loss_barrier = 1e-4 * mean( max(activations - 448, 0)**2 )`  
summed over all 1476 sites where E4M3 rounding is applied. The constant 448 is the E4M3‑representable max. Apply from step 0.

**Why**  
Activation blow‑up beyond the quantisation range was the identified divergence mode. The barrier directly suppresses this instability, making all other loss experiments viable.

**Verification**  
Monitor the 99th percentile activation value per layer every 100 steps. Once the barrier works, these values should stabilise below 440; no sudden jumps that cause NaN losses.

---

#### 5. **PatchGAN + feature matching on E4M3 real** (texture realism, cautious addition)
**Change** (planned)  
A lightweight 5‑layer convolutional discriminator (spectral norm, 64→512 filters, band‑limited to Nyquist/2) seeing the **E4M3‑quantized** real images. Loss: hinge adversarial loss + VGG‑based feature matching (layers relu2_2, relu3_3), weight 0.005. Discriminator learning rate 1e-4, warm‑up 200 steps, ramp to full weight over 500 steps. Train the generator with the same combined loss.

**Why**  
Adversarial pressure can turn “plausible but slightly artificial” pores into genuinely photoreal micro‑texture when the generator is already well‑behaved. Using quantised reals aligns the discriminator’s perception with the model’s own FP8 output, avoiding a mismatch that could cause turbulent textures. The low weight + feature matching minimises GAN‑artifacts.

**Verification**  
After 3 k steps, compute FID (inception‑based, using high‑pass‑filtered patches) between generated and real sharp skin areas. Look for a clear improvement (≥2 FID) while checking that edge halos do not increase.

---

#### 6. **Multi‑scale Laplacian band L1** (moderate gain if simple high‑pass is insufficient)
**Change** (planned – keep as optional)  
Replace the current `high‑pass L1` with an L1 loss on the Laplacian pyramid (4 levels, from 1× to 1/8× resolution), down‑weighting the coarser levels (weights: 1.0, 0.5, 0.25, 0.125). Retain the `local high‑pass energy` as an additional regulariser at the finest level only.

**Why**  
Artifacts like low‑frequency “plasticity” can survive when only one high‑pass band is supervised. The pyramid forces the model to match edge contrast at all scales and often reduces subtle ringing without needing a dedicated anti‑halo hinge.

**Verification**  
Compare the Laplacian error across bands on a test set; consistent reduction in the ½‑ and ¼‑resolution bands indicates better contrast transfer. Visually confirm that transitions from blurred to sharp skin (e.g., cheek to sideburn) stay halo‑free.

---

#### 7. **EMA 0.999 & checkpoint soup** (zero‑cost robustness)
**Changes** (planned – keep both)
- Maintain exponential moving average of weights during fine‑tuning (decay 0.999). At the end, use the EMA weights for evaluation.
- Additionally, take the EMA weights from the last 5–10 checkpoints (every 200 steps), average them, and treat that as the final model.

**Why**  
Both reduce variance and often produce a “smoother” generator that hallucinates less texture in smooth areas. Soup can be done once offline in minutes – an essentially free improvement.

**Verification**  
A/B test EMA‑only vs. souped model on a few video clips. Look for reduced mottle in sky/background and more consistent skin grain from frame to frame.

---

### Explicitly DROP as low value

- **Full learning‑rate sweep (1e‑6..1e‑5)**  
  A full sweep is wasteful. Instead, after the most impactful changes are fixed, run two shorter probes (e.g., 500 steps) at 5e‑6 and 7e‑6 to confirm the current 3e‑6 is not suboptimal. Beyond that, fine‑grain tuning can follow after all other pieces are in place.

- **Anti‑halo/anti‑mottle hinges (planned)**  
  These were conceived when per‑band contrast matching diverged, but the combination of target‑detail masking, multi‑scale Laplacian L1 (or the simple high‑pass L1 with masking), and temporal consistency already suppresses halos and mottle more directly. Adding yet another ad‑hoc hinge would introduce two sensitive hyperparameters (band thresholds, hinge margin) with little marginal benefit.

- **Per‑band contrast matching (already divergent)**  
  Discard – the attempt showed that matching contrast ratios per Laplacian band destabilises the FP8 network. No need to revisit.

- **Dedicated LPIPS/DISTS/CLIP ranking as a training loss**  
  Those metrics are used only for checkpoint **selection** (planned), which is cost‑effective; adding them as an online loss would double the perceptual supervision cost (several hundred ms/step) while the DINOv2‑base loss already captures deep feature similarity. Keep evaluation/ranking but drop as a training loss.

---

By implementing suggestions 1‑3 and 7 first, then cautiously adding 4‑6, you can expect the largest jump in photorealism while staying within a few hours of local compute per trial.

</details>
