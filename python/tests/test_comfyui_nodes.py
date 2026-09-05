import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from .synthetic import synthetic_weights, write_logical_safetensors
from .test_framegen import synthetic_framegen_weights

NODES = pathlib.Path(__file__).resolve().parents[2] / "comfyui" / "ComfyUI-MLX-DLSS" / "nodes.py"


def load_nodes():
    spec = importlib.util.spec_from_file_location("mlxdlss_comfy_nodes", NODES)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ComfyNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_nodes()
        cls.directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls.directory.name)
        cls.renderer_weights = root / "nr.safetensors"
        write_logical_safetensors(cls.renderer_weights, synthetic_weights())
        from safetensors.torch import save_file

        cls.framegen_weights = root / "fg.safetensors"
        save_file(synthetic_framegen_weights(), str(cls.framegen_weights), metadata={"format": "dlssg-framegen-dense-v1"})

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_mappings_and_schemas(self):
        n = self.nodes
        self.assertEqual(set(n.NODE_CLASS_MAPPINGS), set(n.NODE_DISPLAY_NAME_MAPPINGS))
        for cls in n.NODE_CLASS_MAPPINGS.values():
            schema = cls.INPUT_TYPES()
            self.assertIn("required", schema)
            self.assertTrue(hasattr(cls, cls.FUNCTION))

    def test_renderer_node_matches_the_pipeline_and_advances_the_noise_index(self):
        n = self.nodes
        (renderer,) = n.MLXDLSSLoadRenderer().load(str(self.renderer_weights), "cpu", "reference")
        images = torch.from_numpy(np.random.default_rng(0).random((2, 16, 24, 3), dtype=np.float32))
        out, masks = n.MLXDLSSNeuralRendering().render(renderer, images, "standard", 2.0, 1.0, 0.5, 4.0, 1.0, "none", 0.0, 8.0, 7)
        self.assertEqual(tuple(out.shape), (2, 16, 24, 3)); self.assertEqual(tuple(masks.shape), (2, 16, 24))
        direct = renderer.enhance(images[1].numpy(), processing_scale=2.0, colour_strength=0.5, frame_index=8).image
        self.assertLess(float(np.abs(out[1].numpy() - direct).max()), 1e-6)

    def test_renderer_node_takes_an_explicit_skin_mask(self):
        n = self.nodes
        (renderer,) = n.MLXDLSSLoadRenderer().load(str(self.renderer_weights), "cpu", "reference")
        image = torch.from_numpy(np.random.default_rng(1).random((1, 16, 24, 3), dtype=np.float32))
        mask = torch.zeros((1, 8, 12)); mask[:, 2:6, 3:9] = 1          # half-size mask is resampled to the image
        out, used = n.MLXDLSSNeuralRendering().render(renderer, image, "standard", 1.0, 1.0, 1.0, 4.0, 1.0, "skin", 0.0, 8.0, 0, skin_mask=mask)
        self.assertEqual(tuple(used.shape), (1, 16, 24))
        self.assertGreater(float(used.max()), 0.99); self.assertLess(float(used.min()), 0.01)

    def test_framegen_node_interleaves_and_holds_cuts(self):
        n = self.nodes
        (fg,) = n.MLXDLSSLoadFrameGen().load(str(self.framegen_weights), "cpu", "reference")
        bright = torch.full((3, 16, 16, 3), 0.8); dark = torch.zeros((3, 16, 16, 3))
        frames = torch.cat([bright, dark])                                # a hard cut between frames 2 and 3
        out, cuts = n.MLXDLSSFrameGeneration().generate(fg, frames, 4, 0.15, 2)
        self.assertEqual(out.shape[0], 6 + 5 * 3)
        self.assertEqual(cuts, 1)
        seam = out[9:12]                                                  # phases .25 .5 .75 between frames 2 and 3
        self.assertTrue(torch.allclose(seam[0], frames[2], atol=1 / 255))
        self.assertTrue(torch.allclose(seam[1], frames[3], atol=1 / 255))
        self.assertTrue(torch.allclose(seam[2], frames[3], atol=1 / 255))
        self.assertTrue(torch.equal(out[0], torch.from_numpy(n._to_uint8(frames[0]).astype(np.float32) / 255)))

    def test_single_frame_passes_through(self):
        n = self.nodes
        (fg,) = n.MLXDLSSLoadFrameGen().load(str(self.framegen_weights), "cpu", "reference")
        one = torch.zeros((1, 8, 8, 3))
        out, cuts = n.MLXDLSSFrameGeneration().generate(fg, one, 2, 0.15, 4)
        self.assertEqual(out.shape[0], 1); self.assertEqual(cuts, 0)
