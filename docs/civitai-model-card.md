# Neural Re-Detailer (*DLSSDetailer*)

**Fine-tuned weights for the recovered NVIDIA DLSS neural renderer — so its detail actually lands on your footage.**

The DLSS neural renderer was recovered from NVIDIA's libraries and taught to run outside RTX games (upstream MLX-DLSS). But the stock weights add fine texture only to already-sharp stills. **Neural Re-Detailer** is a fine-tune that makes it work where you actually need it: soft AI video, phone footage, upscaled frames. A light finishing pass that puts crispness back.

## What it does

- **Re-details after upscaling** — `--scale` upscales first (Lanczos), then the re-detail pass adds measured high-frequency crispness
- **Works on soft sources** — AI-generated video, phone footage, upscaled stills; where the stock weights do nothing
- **Skin auto-mask** (`--auto-mask skin`) — skin detail lands on face skin; hair and background keep their own texture
- **Temporal mode for video** — reprojects previous output frame to frame, with high-pass history blending to kill static shimmer; scene cuts are held, not blended

## What it does NOT do

Being honest, because you should know what you're downloading:

- **It does not invent objects or change the subject.** No new content, no re-imagining.
- **The fine texture is mostly synthesised.** Measured against ground truth, only **~30% of what it adds aligns with the true lost detail**.
- **This is enhancement, not reconstruction.** A small, measured amount of crispness — not a super-resolution miracle. Use it as a light finishing pass.

## How to use

The weights **auto-download on first use** from Hugging Face ([`bambushu/neural-re-detailer`](https://huggingface.co/bambushu/neural-re-detailer)) — no `--weights` flag needed.

### ComfyUI

Node pack `ComfyUI-MLX-DLSS` (in the fork repo), with example workflows:

- **Neural Re-Detailer — Image Upscale** — upscale + re-detail a still
- **Neural Re-Detailer — Video Upscale** — the same for clips

### CLI

Video:

```sh
mlxdlss-video convert in.mp4 out.mp4 --scale 1.5 --detail-strength 2 --temporal
```

Stills:

```sh
mlxdlss-torch run --input in.png --output out.png --detail-strength 2
```

## Recommended settings

| Setting | Value | Notes |
| --- | --- | --- |
| `--scale` | **1.5-2** | Lanczos pre-upscale before the re-detail pass |
| `--detail-strength` | **2** | the visible default; **1** = subtle / most fidelity-safe, **3** = grain |
| `--temporal` | video | temporal consistency across frames |
| `--hp-history 0.5` | video | removes static shimmer from the temporal path |
| `--auto-mask skin` | faces | keeps skin detail on skin |

## Limitations

- **Best on soft sources and 2x+ upscales.** On already-sharp footage there is little for it to do.
- **Memory:** measured peak RAM is **~1.3 GB at 0.6 MP to ~2.4 GB at 4 MP**.
- **The ComfyUI node buffers the whole sequence** (ComfyUI's batch model), so its RAM is frames x frame size — long or high-res clips can need a lot more.
- **Use the CLI for long clips.** `mlxdlss-video convert` streams one frame at a time and handles any clip length.

## Credits

- **Upstream:** [MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS) by WaveCut ([@iamwavecut](https://github.com/iamwavecut)) — recovered the NVIDIA DLSS neural networks from the vendor libraries so they can run anywhere.
- **These weights:** fine-tuned in the [Bambushu fork](https://github.com/Bambushu/MLX-DLSS) of that project; derived from the vendor libraries and published at [`bambushu/neural-re-detailer`](https://huggingface.co/bambushu/neural-re-detailer).
- **Not affiliated with or endorsed by NVIDIA.** DLSS is NVIDIA's technology; this is an independent recovery and fine-tune of weights extracted from the vendor libraries, which keep the vendor's terms.

## License

Source code is Apache 2.0. **The model weights are derived from NVIDIA's vendor libraries and carry the vendor's terms — do not redistribute them.** Not affiliated with or endorsed by NVIDIA.
