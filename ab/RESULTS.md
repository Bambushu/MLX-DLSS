# DLSSDetailer — measured behaviour

What the fine-tuned re-detail pass actually does, measured against ground truth. Honest positioning:
it is a **detail-enhancement pass that synthesises plausible texture**, not a super-resolution or a
reconstruction of the original detail.

## Recovers, or invents?

Test: take a real frame, degrade it (downscale then upscale, so the native frame is the ground truth),
re-detail it, and correlate the detail the net *adds* with the detail that was truly *lost*.

- Correlation of added detail with true-lost detail: **~0.27–0.28** at every strength, on both a degraded
  real frame and a fresh, clean AI-video frame. So **~70–75% of the added texture is synthesised**, not
  recovered. Strength changes the *amount* of added texture, not the synthesised fraction.
- On perceptual metrics the result still moves toward the true frame at low strength (LPIPS/DISTS improve
  vs the soft input), because plausible texture reads as more natural than mush — but that is not
  reconstruction. On smooth skin, higher strengths visibly paint pore/grain texture the real frame lacks.

Conclusion: treat it as a **stylistic detail-enhancement pass**, honest about synthesising texture.

## Detail strength

| strength | effect |
|---|---|
| 1 | subtle |
| 2 | visible crispness (default) |
| 3 | grain / over-etched |

Fidelity to the true frame is best at low strength; 2 is the visible sweet spot; 3 over-textures.

## Value by upscale factor

The payoff scales with how much detail the upscale destroyed (barista clip, single fine-tune):

| upscale | perceptual gain toward truth | sharpness (MUSIQ) |
|---|---|---|
| 1.5x | small (near break-even) | +4 |
| 2.0x | clear | +8 |
| 3.0x | largest | +13 |

Below ~1.5x it is cosmetic polish; at 2x and above it does real work. Use it when the upscale is >= 2x.

## Upscaler comparison (bake, 12 clips x 3 frames, 1.5x)

Thera (arbitrary-scale) is the best pixel source on fidelity and perceived quality with low halo; Lanczos
second; bicubic third. Learned 2x/4x nets (SPAN, ClearReality, ESRGAN) lose 10+ dB and triple the halo.
The re-detail pass adds ~+3.5 MUSIQ on a clean source. DLSS Super Resolution itself was measured and left
out: without engine motion vectors it loses to plain Lanczos.

## Video / flicker

The temporal path (optical-flow history reprojection + a high-pass history blend) keeps the added detail
*proportionally steadier than the soft source*: measured relative crawl 12% vs 16% (background) and 6.5%
vs 9.2% (face). Flicker is not a problem. `--noise-mode advected` and `--hp-history 0.5–0.7` trim it a
little more.

## Recommendation

Upscale (Lanczos 1.5x or Thera) then re-detail at **detail 2** (or 1 for the most fidelity-safe pass),
weights `dlssnr-ft-real-v2` (or `-v1-crisp` for a touch sharper), `--auto-mask skin` for faces,
`--temporal --hp-history 0.5`. Biggest payoff on 2x+ upscales. The fine-tune is what does the work — the
stock recovered weights are inert on soft input.
