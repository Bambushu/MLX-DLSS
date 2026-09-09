"""``mlxdlss-video``: convert or inspect videos, or compare results in mpv."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from .features import PROFILES
from .pipeline import PRECISIONS, NeuralRenderingPipeline
from .video import DEFAULT_ENCODE_ARGS, PIXEL_FORMATS, ConvertOptions, VideoToolError, compare_command, convert, probe
from .framegen_video import AUDIO_MODES, FrameGenOptions, interpolate_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlxdlss-video", description="Video enhancement through FFmpeg and the recovered neural-rendering transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("convert", help="enhance every frame of a video")
    run.add_argument("input", type=Path); run.add_argument("output", type=Path)
    run.add_argument("--weights", type=Path, help="logical safetensors (from mlxdlss-weights); omit to auto-resolve (MLXDLSS_WEIGHTS / weights dir / MLXDLSS_HF_REPO)")
    run.add_argument("--backend", default="torch", choices=("torch", "mlxdlss"), help="'mlxdlss' streams frames through the Swift Metal runtime (macOS)")
    run.add_argument("--model", type=Path, help="MODEL.dlssmodel for --backend mlxdlss")
    run.add_argument("--mlxdlss", default=None, help="path to the mlxdlss binary (default: PATH or the repository build)")
    run.add_argument("--execution", default="metal-fused", help="mlxdlss execution mode (metal-fused, eager, block-compiled, int8-fast)")
    run.add_argument("--mlxdlss-precision", default="float16", choices=("float16", "float32"))
    run.add_argument("--device", default="auto"); run.add_argument("--precision", default="reference", choices=PRECISIONS)
    run.add_argument("--profile", default="standard", choices=tuple(PROFILES))
    run.add_argument("--processing-scale", type=float, default=1.0)
    run.add_argument("--detail-strength", type=float, default=1.0); run.add_argument("--colour-strength", type=float, default=1.0)
    run.add_argument("--detail-radius", type=float, default=4.0); run.add_argument("--intensity", type=float, default=1.0)
    run.add_argument("--no-degrid", action="store_true", help="keep the fine-tunes' period-4 token grid (default: notched out)")
    run.add_argument("--start-frame", type=int, default=0); run.add_argument("--frames", type=int, default=None, help="stop after this many frames")
    run.add_argument("--batch", type=int, default=1, help="frames per network call (GPU devices)")
    run.add_argument("--pix-fmt", default="rgb24", choices=tuple(PIXEL_FORMATS), help="frame exchange format; rgb48le keeps 16-bit sources")
    run.add_argument("--scale", type=float, default=1.0, help="upscale factor (1.0-4.0) applied before re-detail via a Lanczos pre-pass; 1.5 is the sweet spot, 1.0 = no upscale. Runs first, through an 8-bit intermediate, so pair it with --decode-args on the source not the upscaled frames; for &gt;8-bit or bespoke encodes, upscale separately and omit --scale")
    run.add_argument("--decode-args", default="", help="extra FFmpeg input options, quoted, e.g. \"-vf scale=1280:-2\"")
    run.add_argument("--encode-args", default=None, help=f"FFmpeg output options replacing the default: {' '.join(DEFAULT_ENCODE_ARGS)}")
    run.add_argument("--temporal", action="store_true", help="temporal mode: reprojected history + learned blend (native scale)")
    run.add_argument("--motion", default="flow", choices=("flow", "zero"), help="temporal motion source: OpenCV DIS optical flow or none")
    run.add_argument("--scene-cut", type=float, default=0.3, help="mean luma change that resets the history (0 disables)")
    run.add_argument("--auto-mask", default="none", choices=("none", "skin"), help="skin: per-frame face-parsing mask confines the structure/detail pass to skin (torch backend)")
    run.add_argument("--noise-mode", default="fresh", choices=("fresh", "frozen", "zero", "advected"), help="temporal, torch backend: the 3 noise channels per frame — fresh (vendor), frozen, zero, or advected along the motion")
    run.add_argument("--hp-history", type=float, default=0.0, help="temporal, torch backend: extra history weight on the high-pass band (0-0.7), photometrically gated; reduces re-rolling of invented detail")
    run.add_argument("--mask-floor", type=float, default=0.0, help="structure strength kept outside the auto mask, 0-1")
    run.add_argument("--mask-feather", type=float, default=8.0, help="auto-mask edge softening, gaussian sigma in pixels")
    run.add_argument("--blend-scale", type=float, default=None, help="cap of the learned history blend (default: recovered 0.7397)")
    run.add_argument("--audio", default="copy", choices=("copy", "none"))
    run.add_argument("--overwrite", action="store_true"); run.add_argument("--status-interval", type=float, default=60.0)
    run.add_argument("--ffmpeg", default=None); run.add_argument("--ffprobe", default=None)
    fg = commands.add_parser("framegen", help="generate intermediate frames (DLSS frame generation port): higher frame rate or slow motion")
    fg.add_argument("input", type=Path); fg.add_argument("output", type=Path)
    fg.add_argument("--weights", type=Path, required=True, help="dense frame generation safetensors (from mlxdlss-weights extract-fg)")
    fg.add_argument("--mode", default="fps", choices=("fps", "slowmo"), help="fps: frame rate x factor, same duration; slowmo: same rate, duration x factor")
    fg.add_argument("--factor", type=int, default=2, help="2 doubles (one generated frame per pair), 3 or 4 generate several phases")
    fg.add_argument("--audio", default="copy", choices=AUDIO_MODES, help="copy the audio, stretch it to the slowed video (atempo, pitch kept) or drop it")
    fg.add_argument("--device", default="auto"); fg.add_argument("--precision", default="reference", choices=("reference", "fast"))
    fg.add_argument("--batch", type=int, default=4, help="consecutive frame pairs generated per pass (default 4)")
    fg.add_argument("--scene-cut", type=float, default=0.15, help="mean luma change (0-1) between two input frames that marks a cut: the pair is held (hard cut at the midpoint) instead of blended; 0 disables (default 0.15; clean 12 fps footage measures under 0.1, a cut 0.27+)")
    fg.add_argument("--backend", default="torch", choices=("torch", "mlxdlss"), help="'mlxdlss' streams frames through the Swift Metal runtime (macOS)")
    fg.add_argument("--mlxdlss", default=None, help="path to the mlxdlss binary (default: PATH or the repository build)")
    fg.add_argument("--mlxdlss-precision", default="float16", choices=("float16", "float32"))
    fg.add_argument("--frames", type=int, default=None, help="stop after this many input frames")
    fg.add_argument("--decode-args", default="", help="extra FFmpeg input options, quoted")
    fg.add_argument("--encode-args", default=None, help=f"FFmpeg output options replacing the default: {' '.join(DEFAULT_ENCODE_ARGS)}")
    fg.add_argument("--overwrite", action="store_true"); fg.add_argument("--status-interval", type=float, default=60.0)
    fg.add_argument("--ffmpeg", default=None); fg.add_argument("--ffprobe", default=None)
    show = commands.add_parser("probe", help="print the video stream properties FFprobe reports")
    show.add_argument("input", type=Path); show.add_argument("--ffprobe", default=None)
    compare = commands.add_parser("compare", help="open the original and the processed video side by side in mpv")
    compare.add_argument("original", type=Path); compare.add_argument("processed", type=Path); compare.add_argument("--player", default=None)
    compare.add_argument("--print-only", action="store_true", help="print the mpv command instead of launching it")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            info = probe(args.input, ffprobe=args.ffprobe)
            print(f"{info.width}x{info.height} {info.fps:.3f} fps frames {info.frame_count} codec {info.codec} pix_fmt {info.pixel_format} audio {'yes' if info.has_audio else 'no'}")
            return 0
        if args.command == "compare":
            command = compare_command(args.original, args.processed, player=args.player)
            if args.print_only:
                print(shlex.join(command)); return 0
            return subprocess.call(command)
        if args.command == "framegen":
            from .framegen import FrameGenerator

            generator = None if args.backend == "mlxdlss" else FrameGenerator.from_safetensors(args.weights, device=args.device, precision=args.precision)
            options = FrameGenOptions(
                mode=args.mode, factor=args.factor, audio=args.audio, frame_limit=args.frames,
                decode_args=shlex.split(args.decode_args), encode_args=None if args.encode_args is None else shlex.split(args.encode_args),
                overwrite=args.overwrite, status_interval=args.status_interval,
                backend=args.backend, mlxdlss=args.mlxdlss, mlxdlss_weights=str(args.weights), mlxdlss_precision=args.mlxdlss_precision, batch=args.batch,
                scene_cut_threshold=args.scene_cut,
            )
            result = interpolate_video(args.input, args.output, generator, options, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            where = "mlxdlss metal" if generator is None else str(generator.device)
            print(f"wrote {result.output} ({result.input_frames} -> {result.output_frames} frames, {result.width}x{result.height}, "
                  f"{result.input_fps:.3f} -> {result.output_fps:.3f} fps, {result.output_frames / result.seconds if result.seconds else 0:.2f} fps on {where}, scene cuts {result.scene_cuts})")
            return 0
        pipeline = None
        if args.backend == "torch":
            from .weights import resolve_weights

            pipeline = NeuralRenderingPipeline.from_safetensors(resolve_weights(args.weights), device=args.device, precision=args.precision)
        elif args.model is None:
            raise VideoToolError("--model MODEL.dlssmodel is required for --backend mlxdlss")
        options = ConvertOptions(
            start_frame=args.start_frame, frame_limit=args.frames, batch=args.batch, pixel_format=args.pix_fmt,
            decode_args=shlex.split(args.decode_args), encode_args=None if args.encode_args is None else shlex.split(args.encode_args),
            audio=args.audio, overwrite=args.overwrite, status_interval=args.status_interval,
            temporal=args.temporal, motion=args.motion, scene_cut_threshold=args.scene_cut,
            auto_mask=args.auto_mask, mask_floor=args.mask_floor, mask_feather=args.mask_feather, noise_mode=args.noise_mode, hp_history=args.hp_history,
            backend=args.backend, model_package=str(args.model) if args.model else None, mlxdlss=args.mlxdlss, execution=args.execution, precision=args.mlxdlss_precision,
            **({"blend_scale": args.blend_scale} if args.blend_scale is not None else {}),
            enhance={"profile": args.profile, "processing_scale": args.processing_scale, "detail_strength": args.detail_strength,
                     "colour_strength": args.colour_strength, "detail_radius": args.detail_radius, "intensity": args.intensity, "degrid": not args.no_degrid},
        )
        import math

        if args.scale != 1.0 and not (math.isfinite(args.scale) and 1.0 <= args.scale <= 4.0):
            raise VideoToolError("--scale must be between 1.0 and 4.0")
        import shutil
        import tempfile

        source, scratch = args.input, None
        try:
            if args.scale != 1.0:
                from .video import find_tool

                scratch = Path(tempfile.mkdtemp(prefix="mlxdlss-scale-"))
                source = scratch / "upscaled.mp4"
                # trunc to even dimensions so libx264/yuv420p accepts odd-sized sources
                vf = f"scale=trunc(iw*{args.scale}/2)*2:trunc(ih*{args.scale}/2)*2:flags=lanczos"
                try:
                    subprocess.run([find_tool("ffmpeg", args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                                    "-i", str(args.input), "-map", "0:v:0", "-map", "0:a?", "-vf", vf,
                                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "copy", str(source)], check=True)
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise VideoToolError(f"pre-upscale (--scale {args.scale}) failed: {exc}") from exc
            result = convert(source, args.output, pipeline, options, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
        where = f"{pipeline.device}" if pipeline is not None else "mlxdlss metal"
        print(f"wrote {result.output} ({result.frames} frames, {result.width}x{result.height}, {result.frames / result.seconds if result.seconds else 0:.2f} fps on {where}{', temporal, scene cuts ' + str(result.scene_cuts) if args.temporal else ''})")
        return 0
    except VideoToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
