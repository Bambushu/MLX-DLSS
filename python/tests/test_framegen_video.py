import pathlib
import shutil
import subprocess
import tempfile
import unittest

from mlxdlss.framegen import FrameGenerator
from mlxdlss.framegen_video import FrameGenOptions, atempo_chain, interpolate_video
from mlxdlss.video import VideoToolError, probe

from .test_framegen import synthetic_framegen_weights

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class AtempoTests(unittest.TestCase):
    def test_single_step_in_range(self):
        self.assertEqual(atempo_chain(0.5), "atempo=0.5")
        self.assertEqual(atempo_chain(1.0), "atempo=1")
        self.assertEqual(atempo_chain(2.0), "atempo=2")

    def test_slow_ratios_split_into_half_steps(self):
        self.assertEqual(atempo_chain(0.25), "atempo=0.5,atempo=0.5")
        self.assertEqual(atempo_chain(1 / 3), "atempo=0.5,atempo=0.666667")

    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):
            atempo_chain(0)


def _synthetic_video(path: pathlib.Path, frames: int = 6, fps: int = 10, audio: bool = True) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}:duration={frames / fps}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={frames / fps}", "-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class InterpolateVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = FrameGenerator(synthetic_framegen_weights(), device="cpu")

    def test_fps_mode_doubles_the_rate_and_keeps_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=6, fps=10)
            result = interpolate_video(src, dst, self.generator, FrameGenOptions(mode="fps", factor=2), log=lambda _m: None)
            self.assertEqual(result.input_frames, 6)
            self.assertEqual(result.output_frames, 11)
            info = probe(dst)
            self.assertAlmostEqual(info.fps, 20.0, places=3)
            self.assertTrue(info.has_audio)

    def test_slowmo_mode_keeps_the_rate_and_stretches_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=6, fps=10)
            result = interpolate_video(src, dst, self.generator, FrameGenOptions(mode="slowmo", factor=4, audio="stretch"), log=lambda _m: None)
            self.assertEqual(result.output_frames, 6 + 5 * 3)
            info = probe(dst)
            self.assertAlmostEqual(info.fps, 10.0, places=3)
            self.assertEqual(info.frame_count, 21)
            self.assertTrue(info.has_audio)

    def test_audio_none_drops_the_track(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=3, fps=10)
            interpolate_video(src, dst, self.generator, FrameGenOptions(audio="none"), log=lambda _m: None)
            self.assertFalse(probe(dst).has_audio)

    def test_option_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=2, fps=10, audio=False)
            for options in (FrameGenOptions(factor=1), FrameGenOptions(mode="fps", audio="stretch"), FrameGenOptions(mode="other")):
                with self.assertRaises(VideoToolError):
                    interpolate_video(src, dst, self.generator, options, log=lambda _m: None)


def _cut_video(path: pathlib.Path, fps: int = 10) -> None:
    """Three frames of a bright pattern, a hard cut, three frames of black."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}:duration={3 / fps}",
                    "-f", "lavfi", "-i", f"color=black:size=64x48:rate={fps}:duration={3 / fps}",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)], check=True)


def _decode(path: pathlib.Path):
    import numpy as np

    info = probe(path)
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, info.height, info.width, 3).astype(float)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class SceneCutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = FrameGenerator(synthetic_framegen_weights(), device="cpu")

    def test_gate_off_by_default_reports_no_cuts(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _cut_video(src)
            result = interpolate_video(src, dst, self.generator, FrameGenOptions(audio="none"), log=lambda _m: None)
            self.assertEqual(result.scene_cuts, 0)

    def test_gate_holds_the_seam_instead_of_blending(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _cut_video(src)
            lossless = ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p"]
            result = interpolate_video(src, dst, self.generator, FrameGenOptions(audio="none", factor=4, scene_cut_threshold=0.15, encode_args=lossless), log=lambda _m: None)
            self.assertEqual(result.scene_cuts, 1)
            self.assertEqual(result.output_frames, 6 + 5 * 3)
            frames = _decode(dst)
            a, b = frames[8], frames[12]                   # the last pattern frame and the first black frame
            seam = frames[9:12]                            # phases 0.25, 0.5, 0.75
            self.assertLess(abs(seam[0] - a).mean(), 2.0)  # phase < 0.5 holds A
            self.assertLess(abs(seam[1] - b).mean(), 2.0)  # phase >= 0.5 shows B
            self.assertLess(abs(seam[2] - b).mean(), 2.0)
            clean = frames[1]                              # a generated frame inside the pattern segment still comes from the network
            self.assertGreater(min(abs(clean - frames[0]).mean(), abs(clean - frames[4]).mean()), 0.0)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class MetalBackendTests(unittest.TestCase):
    def test_mlxdlss_backend_generates_frames_when_the_binary_exists(self):
        from mlxdlss.mlxdlss_stream import find_mlxdlss

        try:
            find_mlxdlss(None)
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            weights = pathlib.Path(directory) / "fg.safetensors"
            save_file(synthetic_framegen_weights(), str(weights), metadata={"format": "dlssg-framegen-dense-v1"})
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=4, fps=10, audio=False)
            result = interpolate_video(src, dst, None, FrameGenOptions(backend="mlxdlss", mlxdlss_weights=str(weights), factor=3), log=lambda _m: None)
            self.assertEqual(result.output_frames, 4 + 3 * 2)
            self.assertEqual(probe(dst).frame_count, 10)
