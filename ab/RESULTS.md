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
