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
