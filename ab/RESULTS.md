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

## Neural renderer — BLOCKED on nvngx_dlssnr.dll 310.8.0.0 (not in the public SDK)
