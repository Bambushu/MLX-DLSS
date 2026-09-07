# Step 0 A/B log

## Frame generation (DLSS FG port vs ffmpeg minterpolate) — 2026-09-05, M5, MPS, --precision fast

Protocol: drop odd frames of an H3 800x1440 24fps clip (-> 12fps), regenerate to 24, PSNR of
interpolated frames only against the withheld originals. Baseline = ffmpeg minterpolate
mci/aobmc/bidir/vsbmc.

| clip | DLSS FG mean | DLSS worst | minterpolate mean | minterp worst | DLSS speed | minterp speed |
|---|---|---|---|---|---|---|
| indie_faces (slow, face) | 43.30 dB | - | 40.97 dB | - | 95 fps out | ~15 fps |
| indie_yoga (body motion) | 41.74 dB | 31.22 dB (f169) | 39.66 dB | 24.65 dB | 96 fps out | ~15 fps |

Visual (strip_fg.png, strip_yoga_worst.png, 100% crops): DLSS interp frames hold freckles/eye
detail with no ghosting on the face clip; on the yoga worst frame it smears fast hair and blurs
fingers slightly, minterpolate produces a translucent ghost hand. Source has 0 scene cuts, so the
cut-gate risk is untested here.

Verdict: frame gen is a clear win over the ffmpeg baseline, +2.1-2.3 dB and ~6x faster. Not yet
compared against RIFE/FILM.

## Neural renderer — 2026-09-05, M5, MPS, --precision fast

DLL: nvngx_dlssnr.dll 310.8.0.0 from RankFTW/rhi-repo tag dlssnr-310.8.0 (the NBA 2K27 leak,
165.8MB). SHA-256 e16bcf15… — NOT the checkpoint the port pins (ceb6432f…), but `mlxdlss-weights
all` extracted 649 tensors and decoded clean, so the layout matches.

Single frame (frame 101 of indie_faces, 800x1440):

| setting | time | mean abs change | mean shift |
|---|---|---|---|
| standard | 8.7 s | 4.9/255 | -1.1 |
| natural | 5.3 s | 5.3/255 | -2.8 |
| scale 2 detail 2 | 16.7 s | 4.5/255 | -2.9 |

Visual at 100% and 2x (strip_nr.png, strip_nr_zoom.png): fine grain / pore-like micro-contrast
on skin, freckle edges slightly crisper, slight warm/darker tone. No new structure invented, no
CG sheen. On an already-soft h264 H3 source the gain is modest. Most of the measured change is
tone, not detail.

Video, 24 frames, --temporal: 0.22 fps (109 s), consecutive-frame diff 0.94x the source, so no
added flicker (slightly smoother via history blend). Full 15 s clip would be ~27 min on the M5.

Verdict: usable as a subtle skin-texture / tone pass, not a replacement for LTX-2.5 upscale
(which synthesizes structure). Not yet A/B'd against LTX-2.5 on the same frame — that is a pod
job. Items 2 (skin auto-mask) and 6 (photoreal fine-tune) are what would make it matter.

## Round 2 — 2026-09-05

### Frame gen vs RIFE 4.7 (same withheld-frame protocol, M5 MPS, unbatched single pairs)

| clip | DLSS FG | RIFE 4.7 | minterpolate |
|---|---|---|---|
| faces | 43.30 dB | 43.22 dB (worst 29.5) | 40.97 |
| yoga | 41.74 dB (worst 31.2) | 42.20 dB (worst 31.5) | 39.66 |
| speed 800x1440 | ~95 fps out | ~42 fps out | ~15 fps |

Verdict revised: DLSS FG is a TIE with RIFE 4.7 on quality (RIFE +0.5 dB on motion), ~2x faster.
It is not a quality upgrade over what ComfyUI-Frame-Interpolation already gives us; it is a
faster equivalent. RIFE 4.26 (newer, also in the pack) not tested and may beat both.

### Frame gen across a hard cut
Concat faces|yoga at 12fps, interpolate: the seam frame is a full ghost double-exposure
(strip_cut.png). Scene-cut gate is mandatory before any production use.

### 4x slowmo, yoga hand plant (strip_slow4.png)
Phases .25/.5/.75 coherent, fingers mildly doubled at .25. Usable.

### Renderer on a sharp 2MP Krea2 still (ray_balcony_evening, strip_krea_zoom.png)
4.4 s/frame. `standard` adds believable pores where Krea skin was plastic-smooth — a real
upgrade on a sharp source, the opposite of the soft-H3 result. `--detail-strength 3 --colour 0`
overcooks into orange-peel. `cinematic` = contrast + darker. Renderer wants a sharp input:
run it AFTER upscale, not before, and on Krea2 stills directly.

### Colour-strength sweep at scale 2 (Krea2 stills, sheet3_*.jpg) — 2026-09-05
Mean skin RGB over the face crop:

| still | input | c=0 | c=0.3 | c=0.5 | c=1 (default) |
|---|---|---|---|---|---|
| window_rain | 139 104 82 | 139 104 82 | 138 104 82 | 137 103 82 | 136 103 81 |
| bedroom_golden | 127 89 62 | - | 125 87 60 | 124 86 59 | 121 84 57 |

The colour term darkens/desaturates monotonically: ~2% at 0.5, ~5% at 1 on warm low-key
scenes, less on daylight. Visually 0.3-0.5 keep the scene's warmth and still gain local tone
contrast; 1.0 flattens golden-hour. Detail contribution is identical across the sweep.

**Krea2 still recipe (current):** `--processing-scale 2 --detail-strength 1 --colour-strength 0.5`
(0.3 for warm/low-key scenes; detail 2 for tight close-ups only). ~10 s/still on the M5.

## Roadmap item 1 — scene-cut gate for frame gen — DONE 2026-09-05
Per-pair mean absolute Rec.709 luma change, default threshold 0.15 (`--scene-cut`). Measured:
clean 12 fps pairs max 0.094 (faces, fast head turn), the faces|yoga seam 0.275. On a cut the
generated slots hold A for phases < 0.5 and B from 0.5 (hard cut at the midpoint).
Gate: concat seam held (strip_cut_gated.png, 1 cut reported), faces/yoga PSNR unchanged at
43.30 / 41.74 dB with 0 cuts reported. Unit tests in tests/test_framegen_video.py::SceneCutTests.
Also in the web runner (fixed 0.15). Not applied to `mlxdlss framegen-stream` on the Swift side —
the Python wrapper gates its output too, so the Metal backend is covered.

## Roadmap item 2 — skin auto-mask for the renderer — DONE 2026-09-05
Segmenter: `jonathandinu/face-parsing` (SegFormer, 19 CelebAMask-HQ classes, cached in HF hub),
0.4 s/still on MPS, no torchvision (preprocessing done in torch). Skin = skin/nose/eyes/brows/
ears/mouth/lips/neck, feathered sigma 8 px.

**Routing finding (exp_*.png):** feeding the mask through the control mask's BLUE channel
switches the skin detail OFF (channels 13/14 go to 0 → hp on skin 2.83 vs 3.03 unmasked = input).
Feeding it per pixel into channels 13 AND 14 (the vendor's skin / automatic-mask channels, mask
mode on) keeps full skin detail (3.02) and stops the renderer softening hair/background
(hp outside 4.73 vs 4.47 unmasked, input 4.74). Channel 13 alone is weaker (2.95). So the port
now has a `skin_mask` feature input, not a control-mask hack.

Gate (sheet6_automask_*.jpg, `--processing-scale 2 --colour-strength 0.5 --auto-mask skin`):

| still | hp skin in/unmasked/masked | hp outside in/unmasked/masked | change outside unmasked→masked |
|---|---|---|---|
| window_rain | 2.82 / 2.91 / 2.90 | 4.74 / 4.39 / 4.64 | 2.67 → 1.87 |
| bedroom_golden | 3.45 / 3.70 / 3.64 | 2.36 / 2.33 / 2.26 | 2.38 → 1.46 |

Skin detail kept, hair strands and raindrops no longer smoothed, tone pass still global.
Limitation: the parser is FACE-only (face+neck); chest/arm skin gets the floor value.
Control masks now also work at processing_scale != 1 (resampled to the processing extent).

## Roadmap item 3 — ComfyUI node pack — DONE 2026-09-05
`comfyui/ComfyUI-MLX-DLSS`: 4 nodes (load renderer / neural rendering / load framegen / frame
generation) + 2 example workflows, symlinked into `~/ComfyUI-h3/custom_nodes`, mlxdlss installed
editable into the ComfyUI-h3 venv. Gate: renderer node vs CLI on ray_window_rain (scale 2, c0.5,
auto-mask) = max abs diff 0; framegen node on cut12 = 1 cut, seam frame byte-equal to frame B
(phase 0.5), generated frames within 0.86/255 of the crf-10 CLI file. Unit tests
tests/test_comfyui_nodes.py (synthetic weights). Not yet loaded in a RUNNING ComfyUI — needs a
restart of the :8288 instance; not done unilaterally.

### validate.py framegen — 2026-09-05 18:57
| clip | method | mean PSNR (dB) | worst | gen fps |
|---|---|---|---|---|
| src_faces | dlss | 38.06 | 20.58 | 71.9 |
| src_faces | rife47 | 38.05 | 23.47 | 22.2 |
| src_faces | rife426 | 38.27 | 22.90 | 21.3 |
| src_faces | minterpolate | 38.00 | 21.17 | 7.5 |
| src_yoga | dlss | 36.22 | 25.21 | 70.2 |
| src_yoga | rife47 | 36.91 | 25.48 | 22.3 |
| src_yoga | rife426 | 37.33 | 25.36 | 21.6 |
| src_yoga | minterpolate | 36.77 | 24.59 | 7.4 |

### validate.py renderer — 2026-09-05 19:01
| variant | flicker ratio | hp skin | hp outside | s/frame |
|---|---|---|---|---|
| source | 1.00 | 2.57 | 1.55 | - |
| temporal, c0.5 | 0.96 | 2.55 | 1.47 | 4.6 |
| temporal, c0.5, auto-mask | 0.98 | 2.53 | 1.61 | 4.6 |

## Roadmap item 4 — validation script — DONE 2026-09-05
`scripts/validate.py framegen|renderer` (tables above, appended by the script). Protocol is now
raw-frame PSNR (no h264 round trip), so absolute numbers are ~5 dB lower than the round-1
ffmpeg-psnr numbers; rankings are what matter:
- Frame gen: RIFE 4.26 > RIFE 4.7 ≈ DLSS on the face clip; on body motion RIFE 4.26 leads DLSS
  by 1.1 dB. DLSS is ~3x faster than either RIFE. Item 5's gate (beat RIFE 4.26 by >0.5 dB) is
  therefore a 1.6 dB climb on motion — steep.
- Renderer on soft H3 video (24 frames, temporal): hp on skin unchanged (2.55 vs source 2.57) —
  it adds NO skin detail on this source, only tone; the auto-mask keeps the outside texture
  (1.61 vs 1.47 unmasked, source 1.55). Flicker ratio 0.96-0.98, no shimmer.

## Roadmap item 7 — speed — DONE 2026-09-05 (M5)
- PyTorch/MPS batching: `--batch 4` = `--batch 1` (0.23 fps both at 800x1440); the graph is
  evaluated in bounded chunks, batching buys nothing. One still: network 4.06 s, pre/post 0.1 s.
- Swift/Metal backend built (`swift build -c release`, 136 s). Command Line Tools have no Metal
  compiler; the prebuilt `mlx.metallib` from the `mlx==0.31.1` pip wheel (= the mlx-swift
  checkout version) works — script now takes `MLXDLSS_METALLIB`.
- Metal vs PyTorch(fast), same still: network 2.5 s vs 4.1 s; output mean abs diff 0.28/255.
- **Temporal video, 24 frames 800x1440: Metal 3.85 fps vs PyTorch 0.22 fps (17x)**, output within
  0.93/255 of the torch run, flicker ratio 0.96 vs 0.97. A 15 s clip: ~1.5 min instead of 27.
- Frame gen: Metal 136 fps out vs 77 fps (torch) on the 15 s clip; cut gate applies (1 cut on cut12).
- ComfyUI: `MLXDLSSNeuralRenderingMetal` node (temporal video through the Metal stream),
  3.98 fps on the same 24 frames, within 0.96/255 of the CLI. No skin mask on the Metal path yet.

## Roadmap item 6 — photoreal fine-tune — IN PROGRESS 2026-09-05
**Feasibility (the big one):** the recovered graph trains AS-IS. Straight-through estimator on
every `e4m3_round_trip` (`v + (round(v) - v).detach()`), `MLXDLSS_TORCH_CHUNK_TOKENS=0` to avoid
the in-place chunk writers, logical weights flipped to `requires_grad`: finite non-zero
gradients on 508/649 tensors (the 141 without are attn_scale/attn_bias/attention_scalar/
blend_scale constants). 256x256 crop fwd+bwd 2.0 s, 3.9 GB on the M5. No reimplementation of the
attention blocks needed — "weeks" became "days".

`scripts/finetune.py`: self-supervised soft→sharp pairs (sharp still = target; input = downscale
0.5-0.8 + resample + JPEG q28-60 + blur σ0.4-1.0), skin-biased 256 crops, skin mask in channels
13/14, torch composition `soft + 0.25*half(head[:3])`. Dataset: 402 face stills (skin ≥ 5%) from
~/ComfyUI/output + feed-final, 12 held out.

Smoke run (20 steps, plain L1 + hp L1): the stock net at scale 1 LOWERS PSNR vs the soft input
(36.0 → 30.7, the colour prior) and the fine-tune snapped to identity (36.0, hp unchanged) —
pixel losses on hallucinated texture regress to the mean. Loss changed to low-pass L1 + 0.5 hp
L1 + 4.0 local high-pass ENERGY match (alignment-free texture amount). run1: 1500 steps, batch 2,
lr 1.5e-5, ~2 s/step.

### run2 data (Mike: "no krea2 stills") — 2026-09-05
- FFHQ-1024 (gaunernst/ffhq-1024-wds) 12 shards = 12k real faces, LSDIR (danjacobellis) 10
  parquet shards = 1745 images ≥768px, RealSR V3 (eval/calibration only). `scripts/fetch_datasets.py`,
  data at ~/mlx-dlss-data (outside the repo).
- FFHQ-1024 is NOT uniformly sharp: many are upscaled Flickr crops. Sharpness metric
  (mean |L − blur1| on the centre 512) p50 1.44; `--min-sharp 1.6` keeps the pore-level ~45%
  (2821 of 6000). LSDIR median 6.5, 99% pass.
- Degradation calibrated on RealSR: real DSLR x2 pairs keep hp ratio 0.51 (x3: 0.33) of the
  sharp image; the first synthetic degrade kept 0.73 (too mild). Now downscale 0.35-0.65 +
  JPEG 25-55 + blur σ0.5-1.3 → 0.60 (p10 0.43, p90 0.76).
- run2: dataset2 = 2821 FFHQ + 1745 LSDIR, 24 held out, 3000 steps, lr 1e-5, batch 2.

### run2 outcome (real photos, energy loss) — stopped at step 2000
| step | PSNR out (soft 31.15) | hp out (soft 2.51, target 6.49) |
|---|---|---|
| 0 (stock) | 28.36 | 2.55 |
| 500 | 30.65 | 4.51 |
| 1000 | 30.15 | 4.97 |
| 1500 | 29.92 | 4.85 |
| 2000 | 27.34 | 4.63 |

Visual (strip_run2_500/1500.png): on the H3 face the fine-tune is a clear win at 500-1500
(freckles crisp, colour intact). On a RealSR real photo, step 1500 over-sharpens text into halos
and drops PSNR below the soft input (27.2 vs 29.3). The local-energy loss buys texture AMOUNT,
not structure; it plateaus at ~4.9 and then degrades. Pixel-loss-only training cannot do more.

### run3 — DINOv2 perceptual loss (launched 2026-09-05 ~21:10)
`--w-energy 2.0 --w-dino 1.0` (1 − cosine on DINOv2-base patch tokens, model from the HF cache,
no torchvision), plus low 1.0 / hp 0.5; lr 1e-5, 3000 steps, dataset2, from stock weights.
~2.3 s/step alone. Gate: RealSR PSNR must stay ≥ the soft input while hp climbs; H3 face judged
by eye at 100%/2x.

### run3 outcome — DINOv2 perceptual loss — DONE 2026-09-05
| step | held-out PSNR (soft 31.15) | hp (soft 2.51, target 6.49) | RealSR real pair PSNR (LR 29.30, stock 27.72) |
|---|---|---|---|
| 500 | 31.14 | 4.53 | 29.37 |
| 1500 | 31.32 | 4.40 | **29.39** |
| 2000 | 31.27 | 4.67 | – |
| 2500 | 31.02 | **5.25** | 28.72 |
| 3000 | 31.35 | 4.80 | 28.93 |

Gate passed: PSNR at/above the soft input while high-pass roughly doubles; no halos on real
text (strip_run3_final.png). H3 face at 100%: lashes, brows, freckle edges crisp, skin clean at
every checkpoint; 2500 has the most texture, 1500 the best fidelity.
**Shipped as `weights/dlssnr-ft-real-v1.safetensors` (= step 1500) and
`dlssnr-ft-real-v1-punchy.safetensors` (= step 2500)**; Metal package
`weights/NeuralRendering-ft-real-v1.dlssmodel` (`mlxdlss-weights mlx`), Metal output within
0.58/255 of torch. Temporal video (24 H3 frames, Metal): 4.5 fps, flicker ratio 1.04 vs stock 0.96
(slight, no visible shimmer at 100%).
On an already-sharp Krea2 still at the scale-2 recipe the fine-tune adds pores but reads slightly
peppery (strip_run3_krea.png) — for sharp Krea2 stills keep the STOCK weights; the fine-tune is
for soft sources (H3 / AI video / phone footage). Not measured: other H3 clips, non-face video.

### v1 on five more H3 clips (24 frames each, Metal temporal, 2026-09-05) — ab/v1test/
| clip | content | hp skin src→stock→v1 | hp outside src→stock→v1 | static-pixel temporal diff src/stock/v1 | moving-pixel diff src/stock/v1 |
|---|---|---|---|---|---|
| mmh3 | rain-window portrait | 2.98→3.09→3.85 | 2.63→2.67→3.23 | 0.39/0.43/0.50 | 6.6/6.6/8.5 |
| demo_v4 | glamour, jewellery | 4.82→4.59→6.12 | 6.29→6.03→7.43 | 0.26/0.41/0.41 | 8.9/8.2/11.0 |
| loco | locomotive, no face | – | 6.25→6.42→7.74 | 0.41/0.75/0.52 | 9.3/6.7/9.9 |
| golf | presenter, dark skin | 4.58→4.98→5.40 | 2.78→2.81→3.18 | 0.24/0.42/0.45 | 13.2/12.5/13.9 |
| yoga | body, low light | – | 4.41→3.90→4.71 | 0.44/0.59/0.59 | 8.2/6.2/8.6 |

v1 adds 20-30% high-pass detail on every clip (faces, jewellery, machinery, fabric) and keeps
colour where stock greys/darkens. The raw flicker ratio (1.10-1.25) is NOT shimmer: in static
pixels v1's temporal difference equals stock's (≤0.6/255); the extra comes from moving pixels,
i.e. sharper detail travelling with the subject. Strips strip_*.png, side-by-side videos
mmh3_sbs.mp4 / golf_sbs.mp4 (source | stock | v1).

## Overnight queue 2026-09-06 07:41–13:04 (scripts/overnight.sh)
### run4 — run3 recipe, 6000 steps, crop 288, cosine decay (held-out re-cropped at 288, so compare within run)
| step | PSNR (soft 30.35) | hp (soft 2.40, target 6.42) |
|---|---|---|
| 1000 | 30.47 | 4.25 |
| 2000 | 30.76 | 4.23 |
| 3000 | 30.45 | 4.77 |
| 4000 | 30.55 | 4.61 |
| 5000 | 30.58 | 4.62 |
| 6000 | 30.55 | 4.66 |
Converged by 4000. H3 eyes at 100%: marginally crisper lashes/brows than v1, cheek equally clean.
RealSR real pair: 28.89 dB at 4000 and 6000 (v1 29.39, LR 29.30) — slightly more sharpening
than the real photo warrants. Shipped as **`weights/dlssnr-ft-real-v1-crisp.safetensors`** (step
4000), replacing the run3-2500 "punchy" file (28.72 on RealSR, grainier).

### run5 — ablation, DINO 2.0 / energy 1.0, 2500 steps
PSNR 31.47→31.48 (best of any run), hp 3.23→3.26 (half of run3's). RealSR 29.47. Visibly the
softest fine-tune: leaning on the perceptual term converges to a gentle, faithful sharpener.
Answer: energy 2 / DINO 1 (run3/run4) is the right balance; not shipped.

### Metal tails through v1
faces 15 s (362 fr) 4.9 fps, long50s 5.7 fps, golf 5.1 fps, cut12 4.7 fps. No drift over the 15 s
take (|v1−src| 1.41 / 1.52 / 1.56 at start / middle / end, flicker ratio 1.05–1.14). The hard cut
(luma jump 0.275 < the renderer's 0.3 reset threshold) did NOT ghost: frame after the cut within
2.2/255 of the source — the learned history blend rejects mismatched history by itself.

### v1 dial sweep (overnight/sweep_*.jpg)
v1 was trained at scale 1: **scale 1 is its sweet spot; scale 2 is SOFTER with v1** (opposite of
stock). Detail 2 adds grain on cheeks; colour 0.5 vs 1 barely differs (v1 learned colour
fidelity). **v1 recipe: `--processing-scale 1 --detail-strength 1 --colour-strength 1
--auto-mask skin`.** Stock stays scale 2 / colour 0.5 for sharp Krea2 stills.

### v1 vs v1-crisp on six 3-5 s clips (2026-09-06, ab/clips5/, Metal temporal)
| clip | frames | hp src / v1 / crisp | flicker v1 / crisp | static-px diff src / v1 / crisp |
|---|---|---|---|---|
| face_rain | 120 | 2.71 / 3.39 / 3.32 | 1.19 / 1.17 | 0.48 / 0.59 / 0.53 |
| face_glam | 96 | 6.56 / 7.77 / 7.62 | 1.27 / 1.21 | 0.26 / 0.42 / 0.41 |
| face_kitchen (800x448) | 96 | 8.94 / 10.39 / 10.78 | 1.18 / 1.19 | 0.59 / 0.71 / 0.69 |
| nf_loco | 96 | 7.57 / 9.38 / 9.88 | 1.12 / 1.17 | 0.47 / 0.62 / 0.60 |
| nf_car | 72 | 5.41 / 7.38 / 7.73 | 1.27 / 1.31 | 0.76 / 1.14 / 0.97 |
| nf_kitchen wide | 96 | 5.49 / 6.45 / 6.81 | 1.13 / 1.16 | 0.69 / 0.83 / 0.78 |

Faces: v1 and crisp equal (crisp −2% hp, slightly lower static noise). Non-faces: crisp adds
5-6% more detail than v1 (window reflections, plants, raindrops, machinery) with lower static
noise on every clip. **crisp (run4 step 4000) becomes the default fine-tune**; v1 (run3 1500)
kept as the conservative option. Both beat the source on all six at 100%.

## Panel plan B — noise-channel coherence (2026-09-06, ab/noise/)
Crisp weights, torch temporal, 24 frames. Feature channels 0-2 = fresh (vendor) / frozen / zero /
advected-along-flow noise:
| clip | mode | hp f12 | flicker | static-px | moving-px |
|---|---|---|---|---|---|
| face_rain | fresh / frozen / zero / advected | 2.91 / 2.91 / 2.90 / 2.92 | 1.19 / 1.19 / 1.18 / 1.19 | 0.49 / 0.48 / 0.48 / 0.49 | 8.9 / 8.9 / 8.8 / 8.8 |
| nf_car | fresh / frozen / zero / advected | 9.08 / 9.09 / 9.05 / 9.09 | 1.38 / 1.37 / 1.37 / 1.38 | 0.95 / 0.93 / 0.93 / 0.97 | 17.6 / 17.6 / 17.6 / 17.6 |
**No effect at all** — the network (stock or fine-tuned) barely uses the noise channels. The moving-
pixel flicker is per-frame hallucination + codec jitter, not noise. `--noise-mode` stays as a
documented no-op experiment. Next: band-split history blend (more history on the high-pass).

## Panel plan A — degradation v2 calibration (2026-09-06)
Against 31 real H3 frames (ab/ft/h3frames): skin high-pass 2.28 (p10 1.26, p90 3.75) vs FFHQ
sharp targets 3.05 → H3 keeps ~75% of the sharp high-pass, WIDE spread up to near-sharp. Old v1
degrade inputs: 1.48 (softer than H3, opposite of the panel's premise). v2 after tuning: 1.75
(p10 0.81, p90 2.89), RealSR ratio 0.82 (p10 0.46, p90 1.0): brackets H3 from codec-crushed to
clean. Mixture: 20% clean, 10% phone noise, 20% JPEG q35-75, 50% x264 CRF 22-36 4:2:0 (30% double
encode); downscale 0.55-1.0; blur 0-0.7 AFTER the codec; sensor noise before it in 50%.
Skin mask now computed on the degraded crop. run6 = fine-tune from crisp, 3000 steps, cosine, lr 8e-6.

### B part 2 — band-split high-pass history blend (`--hp-history`, torch temporal)
Vendor blend on the low-pass; high-pass pulled toward the reprojected previous OUTPUT by α,
gated by the photometric error of the reprojected previous INPUT (full trust < 1.5/255, none > 15/255).
| clip | variant | hp f12 | flicker | static-px | moving-px |
|---|---|---|---|---|---|
| face_rain | vendor / α0.4 / α0.7 | 2.91 / 2.85 / 2.77 | 1.19 / 1.14 / 1.10 | 0.49 / 0.44 / 0.41 (source 0.43) | 8.9 / 8.6 / 8.4 |
| nf_car | vendor / α0.4 / α0.7 | 9.08 / 8.83 / 8.64 | 1.38 / 1.34 / 1.33 | 0.95 / 0.89 / 0.86 (source 0.71) | 17.6 / 17.3 / 17.2 |
Static-pixel shimmer on the face drops BELOW the source at α0.7 for a 5% detail cost; moving
pixels barely change because the gate (correctly) releases where content moves. Modest, real,
free. Recommendation: `--hp-history 0.5` for video. Default stays 0 (vendor parity). Metal path
does not have it yet.

## Panel plan A result — run6 (degrade v2, mask on degraded, from crisp, 3000 cosine) = "v2" — 2026-09-06
Held-out (v2 degrade, incl. clean inputs): PSNR crisp 34.57 → v2 36.47 (soft 38.58); hp 3.30 → 3.04
(target 3.63). H3 face skin hp: input 2.08, crisp 2.69, v2 2.78 (step 1500) — detail KEPT on soft
H3. RealSR real pair: crisp 28.89 → v2-1500 29.35 (LR 29.30) — over-sharpening of real photos fixed.

Six 3-5 s clips, Metal temporal (ab/clips5/*_crisp_vs_v2.mp4):
| clip | hp src / crisp / v2 | flicker crisp / v2 |
|---|---|---|
| face_rain | 2.71 / 3.32 / 3.06 | 1.17 / 1.15 |
| face_glam | 6.56 / 7.62 / 7.06 | 1.21 / 1.14 |
| face_kitchen | 8.94 / 10.78 / 9.36 | 1.19 / 1.07 |
| nf_loco | 7.57 / 9.88 / 9.50 | 1.17 / 1.17 |
| nf_car (already sharp-ish) | 5.41 / 7.73 / 5.59 | 1.31 / 1.03 |
| nf_kitchen | 5.49 / 6.81 / 6.10 | 1.16 / 1.11 |
v2 is the HONEST model: it adds ~as much as crisp on soft faces, leaves already-sharp content
(the car) nearly alone, and flickers less (car 1.31 → 1.03). Crisp is the PUNCHY model: more bite
everywhere, incl. content that did not need it, more flicker. Shipped both:
`weights/dlssnr-ft-real-v2.safetensors` + `NeuralRendering-ft-real-v2.dlssmodel`.

## Plan C stability ablation (2026-09-06, 260 steps each from v2, lr 3e-6, same samples)

| run | terms | skipped | loss@250 | PSNR out | hp out |
|---|---|---|---|---|---|
| control | energy 2.0 | 0 | 0.222 | 35.36 | 3.14 |
| band | Laplacian band L1 | 0 | 0.227 | 34.01 | 2.83 |
| bandvar | band + variance-sqrt contrast x4 | 55 | 0.821 | 11.95 | 3.99 |
| hinges | band + halo + mottle | 0 | 0.226 | 33.95 | 2.77 |
| all | band + MAD contrast x4 + hinges | 0 | 0.887 | 11.62 | 3.93 |
| hingesE | band + hinges + energy 2.0 | 0 | 0.244 | 33.64 | 3.10 |
| all1 | band + MAD contrast x1 + hinges | 0 | 0.241 | 33.98 | 2.91 |

The per-band contrast term is the destabiliser: as variance-sqrt it NaNs (sqrt'(v+1e-6)=500 on flat
skin), as mean absolute deviation at x4 it still runs the low-pass term from 0.007 to 0.24 (tone
runaway, no NaN). Band L1 and both hinges are harmless. **run7 = hingesE config** (band + halo +
mottle over the proven energy 2.0, `--lap-terms band,halo,mottle`); D and E inherit it. hp-cost of
the hinges vs control: ~1%.

### Why plan C kept diverging: activation blow-up, not weights (2026-09-06)

run7 (hingesE config) collapsed between step 600 and 625 with FINITE gradients (first non-finite
grad at 689). Weight deltas step500→1000 are ≤ 1 fp16 ulp per element, yet the step-1000 weights
render garbage (PSNR 12 vs input). Spying every E4M3 rounding site: v2 / step500 peak |activation|
≈ 100; step1000 peaks at 5138 with 667 of 1476 sites saturated at the E4M3 max (448). The gate
multiplies in fp16 and the round trip clamps overflow silently, so the network keeps a finite loss
while producing nonsense. Nothing in the loss kept activations inside the FP8 envelope; a loss that
pushes contrast (energy + lap) walks them off the cliff, faster the harder it pushes (bandvar x4
~150 steps, MAD x4 ~200, hingesE ~600).

Fix: `--w-act` (default 10) = at every E4M3 site, mean((|v|-256)+/256)^2, summed. Reads 0.0 on v2
(amax 89) and 1037 on the broken checkpoint. run7 relaunched from v2 with hingesE + barrier; the
`act`/`amax` columns are in the log. Diverged run kept at ab/ft/run7_diverged.

**2026-09-06 21:05 the Mac hard-rebooted** with the dense barrier (an extra activation-sized tensor
retained at all 1476 sites) plus the test suite running beside training. Now: barrier on every
20th site, crop 256, a memory watchdog in the script (`--min-free-pct 12 --max-swap-gb 16` →
save + exit 3), and nothing else heavy beside a training run. Measured healthy regime for run7:
trainer footprint 12 GB, free 67%, swap 2 GB, 2.6 s/step.

### Panel round 2 (2026-09-06 evening, docs/research/best-weights-panel.md: MiniMax M3, Kimi K3, DeepSeek V4 Pro)
Taken: lr sweep narrowed to 3e-6 / 6e-6 / 1e-5 + late-blocks (1e-6 dropped); EMA with warm-up decay
(0.1→0.999) and soups of the last checkpoints; IQA metrics are for SELECTION only, never a loss;
a real H3 held-out set (ab/ft/h3frames + ab/clips12) is the arbiter. To audit before the final run:
(a) the fp16 save vs the live fp32 model (eval the saved checkpoint, compare with the logged eval),
(b) the Metal package (.dlssmodel) vs torch on the same frame for a fine-tune, so the fine-tune is
not lost to E4M3 packing, (c) which of the 141 frozen tensors matter. Deferred: bigger crops (288
already swaps on 51 GB), real H3 crops with no reconstruction loss (needs the GAN), training
through the history blend (after plan D), directional edge-annulus halo term (C says drop hinges;
decide on run7's eval). Panel is split on late-blocks-only and on the GAN; both stay in the plan as
measured runs.

### Blind A/B set (ab/clips12, 2026-09-06, 72 frames each, long side ≤1440, crf 10)
Mike's pick: recent production H3 renders only — nine indiewalsh takes (first, comesup, coffee, gym,
walk, grwm, night, car, smoothie) + bloomrest sleeping close-up (2026-09-06), portfolio marcus
(male), barista café (wide). Rejected: old/upscaled/stock material (eros HiRes, casino, airport,
4K terrace, virtual-camera reel) — "I don't want to upscale or DLSS5 it".

### dataset3 (2026-09-06 night, `--min-sharp 1.6 --min-side 768`)
| source | on disk | sharp stills |
|---|---|---|
| FFHQ-1024 (70 shards) | 70000 | 28386 |
| LSDIR (40 shards, ≥768) | 7510 | 7451 |
| Flickr2K | 2650 | 2463 |
| DIV8K (≤3072 px) | 1500 | 1254 |
| **total** | | **39554** (dataset2 was 6579) |

### Chain C→D→E from v2 (2026-09-07 night, crop 256, dense FP8 barrier, held-out 32 crops)
| run | loss | final PSNR | final hp | note |
|---|---|---|---|---|
| v2 (start) | — | 35.90 | 2.81 | |
| run7 = C | band L1 + halo/mottle hinges over energy 2.0, 3000 steps | 35.87 | 2.79 | flat from step 500; a regulariser, not a detail gain |
| run8 = D | + synthetic-flow temporal 1.0, 2500 steps | 35.77 | 2.50 | -11% hp for consistency; first attempt with the sparse barrier hit 500-867 peaks and was discarded |
| run9 = E | + band-limited GAN 0.005 + FM, 2000 steps | 35.62 | 2.57 | the only term that moved hp UP (2.50→2.62 at step 500), oscillates |
Verdict pending rank.py (MUSIQ/TOPIQ/LPIPS/DISTS on H3 frames + RealSR) and the blind A/B; hp
energy alone says v2/crisp still lead on detail.

### Hyper-parameter sweep (2026-09-07 morning, 800 steps from v2, plan-C loss + dense barrier, EMA 0.999 warm-up)
| run | PSNR | hp | |
|---|---|---|---|
| control lr 3e-6 | 36.03 | 2.78 | |
| **lr 6e-6** | **36.10** | **2.83** | best on both axes |
| lr 1e-5 | 36.08 | 2.81 | |
| late blocks only (35-70), 3e-6 | 36.01 | 2.76 | worst; the panel's "within noise" was right |
Final-run lr = 6e-6 unless rank.py disagrees.

### Audits before the final run (2026-09-07 08:28)
- (a) fp16 save vs live model: saved lr6e6 step-800 evaluates PSNR 36.12 / hp 2.84 vs logged 36.10 / 2.83 — nothing lost at save.
- (b) Metal package vs torch, H3 frame, no mask: torch↔Metal 51.1 dB (stock), 47.9 dB (v2); fine-tune effect vs stock 30.9 dB on BOTH paths — the .dlssmodel keeps the fine-tune through FP8 packing.
- (c) frozen 141/649 tensors = attn_scale 70, attn_bias 62, attention_scalar 8, blend_scale 1: recovered constants with no gradient path; nothing trainable is excluded.
- pyiqa metrics must run on CPU (MPS lacks non-divisible adaptive pooling); rendering stays on MPS.
