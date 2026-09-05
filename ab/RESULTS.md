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
