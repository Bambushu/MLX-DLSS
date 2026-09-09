# MLX-DLSS

> [!TIP]
> **A note for the NVIDIA reader.** This port was worked out on a laptop and on
> GPU instances rented by the hour, some of which even booted. A pair of DGX
> Sparks would have replaced the rentals and would have a steady job here:
> experiments like this one, and the pet projects queued behind it. Hit me up on
> X: [@WaveCut](https://x.com/WaveCut).

Upstream by [@WaveCut](https://github.com/iamwavecut/MLX-DLSS): NVIDIA's two DLSS neural
networks — the DLSS 5 neural renderer and the frame generator — recovered from the vendor
libraries and re-implemented to run on stills and video: Apple Silicon first (MLX/Metal), any
machine through PyTorch, plus a Core ML export. Nothing proprietary ships with this repo: you
supply your own `nvngx_dlssnr.dll` / `libnvidia-ngx-dlssg.so` and the weight tool extracts the
tensors locally.

## This fork (Bambushu/MLX-DLSS): Neural Re-Detailer (*DLSSDetailer*)

> [!NOTE]
> **This is not magic.** Neural Re-Detailer is a minor detailer and upscaler. It does not invent
> objects or scene content and it does not change the subject — it adds a small, measured amount
> of crispness. The fine texture is *mostly synthesised*: measured against ground truth, only
> ~30% of what it adds aligns with the true lost detail (see
> [`ab/RESULTS.md`](ab/RESULTS.md)), so treat it as enhancement, not reconstruction. Use it as a
> light finishing pass, not a super-resolution miracle. The gain is real but modest; it does the
> most on soft, upscaled footage.

- **Fine-tuned renderer weights**: the stock weights add detail only to already-sharp stills;
  the fine-tunes (`scripts/finetune.py`) also work on soft sources (AI video, phone footage).
  Published at [`bambushu/neural-re-detailer`](https://huggingface.co/bambushu/neural-re-detailer)
  and **auto-downloaded on first use — no `--weights` needed** (`MLXDLSS_HF_REPO` overrides the
  repo, set it to `""` to disable the fetch).
- **Skin auto-mask** (`--auto-mask skin`): face parsing puts skin detail on skin; hair and
  background keep their own texture.
- **Video niceties**: `--scene-cut` holds hard cuts instead of blending them;
  `--hp-history 0.5` removes static shimmer at ~5% detail cost.
- **Prebuilt Metal backend**: runs even on Command Line Tools; ~17x the PyTorch path on video.
- **ComfyUI node pack** (`comfyui/ComfyUI-MLX-DLSS`): the renderer, frame-gen, and Upscale nodes + `example_workflows/`.
- **`scripts/validate.py`**: withheld-frame PSNR vs RIFE/minterpolate, sharpness and flicker
  checks — every measured number lives in [`ab/RESULTS.md`](ab/RESULTS.md).

![Neural rendering: input, defaults, processing scale 2 with detail 2](docs/assets/neural-rendering-control.png)

https://github.com/user-attachments/assets/ce94f426-910b-4556-bdf9-662cbdd5933a

## Quickstart

- Python 3.10+ (PyTorch installs as a dependency); `ffmpeg`/`ffprobe` in `PATH` for video.
- Metal backend: macOS 14+, full Xcode (Swift 6.2), CMake, Ninja. Optional extras:
  `pip install './python[coreml]'` (fixed-extent export), `'./python[mask]'` (skin mask).

```sh
python -m pip install './python[web,video]'
swift build -c release && scripts/prepare-mlx-metallib.sh "$(swift build -c release --show-bin-path)"  # macOS, Metal; re-run after clean builds
mlxdlss-video convert in.mp4 out.mp4 --scale 1.5 --detail-strength 2 --temporal  # upscale 1.5x, re-detail, temporal
mlxdlss-torch run --input in.png --output out.png --detail-strength 2            # a still
```

The re-detail pass: `--scale` upscales (Lanczos) first; strength 1 is subtle, 2 is the visible
default, 3 grains. Recipe for soft footage: processing scale 1, detail 2, `--auto-mask skin`,
`--hp-history 0.5`; fine-tune `dlssnr-ft-real-v2` for video, `v1-crisp` for stills. Memory: the
CLI streams frames (~1.3–2.4 GB peak, 0.6–4 MP, any clip length); the ComfyUI node buffers the whole sequence.
Full CLI surface: `--help`.

`mlxdlss-web` opens a local front end (before/after image slider, video effect chain, job
queue). Upstream weights still extract from your own DLLs, e.g. `mlxdlss-weights all
nvngx_dlssnr.dll weights/` — recovery details in [docs/recovery-notes.md](docs/recovery-notes.md).

## ComfyUI

`comfyui/ComfyUI-MLX-DLSS` wraps both networks: the renderer with the skin auto-mask, frame
generation with the scene-cut gate, and Upscale nodes that pair a resampler or upscale model
(pixels) with the renderer (detail); `example_workflows/` included. Measured on twelve
production clips: Thera is the best pixel source for the re-detail pass, Lanczos (the default)
is second; learned upscalers (SPAN, Real-ESRGAN…) lose 10+ dB to ground truth and triple the
halo — the wrong input. See [`ab/RESULTS.md`](ab/RESULTS.md).

## What it is not

Not real-time in games, not a drop-in DLSS, not affiliated with or endorsed by NVIDIA. DLSS
Super Resolution was measured and deliberately left out: without engine motion vectors it loses
to plain Lanczos ([docs/super-resolution.md](docs/super-resolution.md)).

## Documentation

- [Frame generation](docs/frame-generation.md) — graph, verification, speed.
- [Super resolution](docs/super-resolution.md) — measured, not ported.
- [Embedding guide](docs/embedding.md) — the Swift API.
- [Recovery notes](docs/recovery-notes.md) — package format, recovered graph, measured errors
  (0.005 MAE renderer, 59.9 dB frame generator vs the NVIDIA DLL).
- [`ab/RESULTS.md`](ab/RESULTS.md) — the fork's measurements, including the ~30% detail
  alignment figure.
- [Research notes](docs/research/), [SECURITY.md](SECURITY.md), [CONTRIBUTING.md](CONTRIBUTING.md), [NOTICE](NOTICE).

Development: `scripts/verify.sh` runs the Swift and Python test suites plus the public-tree audit.

## License

Apache License 2.0 for the source. Model packages built from vendor libraries keep the vendor's
terms — do not redistribute them.
