import os
import unittest

import numpy as np

from mlxdlss.automask import SKIN_LABELS, SkinMasker, feather, skin_mask_with_floor
from mlxdlss.features import half, make_features
from mlxdlss.pipeline import NeuralRenderingPipeline

from .synthetic import synthetic_weights


class SkinMaskTests(unittest.TestCase):
    def test_floor_lifts_the_outside(self):
        skin = np.zeros((4, 6), dtype=np.float32); skin[1:3, 2:5] = 1
        mask = skin_mask_with_floor(skin, floor=0.2)
        self.assertEqual(mask.shape, (4, 6))
        self.assertAlmostEqual(float(mask[2, 3]), 1.0, places=6)
        self.assertAlmostEqual(float(mask[0, 0]), 0.2, places=6)

    def test_floor_and_shape_validation(self):
        with self.assertRaises(ValueError):
            skin_mask_with_floor(np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            skin_mask_with_floor(np.zeros((2, 2), dtype=np.float32), floor=1.5)

    def test_skin_mask_writes_channels_13_and_14_per_pixel_and_enables_mask_mode(self):
        color = np.full((8, 8, 3), 0.5, dtype=np.float32)
        skin = np.zeros((8, 8), dtype=np.float32); skin[2:6, 2:6] = 1
        features = make_features(color, skin_mask=skin, local_structure_strength=0.75)
        self.assertEqual(features[4, 4, 13], half(0.75)); self.assertEqual(features[4, 4, 14], half(0.75))
        self.assertEqual(features[0, 0, 13], half(0.0)); self.assertEqual(features[0, 0, 14], half(0.0))
        self.assertEqual(features[0, 0, 12], half(1))                       # mask mode: structure channel on
        plain = make_features(color, local_structure_strength=0.75)
        self.assertEqual(plain[0, 0, 13], np.float32(-1))                    # without a mask the channels stay disabled
        with self.assertRaises(ValueError):
            make_features(color, skin_mask=np.zeros((4, 4), dtype=np.float32))

    def test_feather_softens_edges_and_keeps_range(self):
        hard = np.zeros((32, 32), dtype=np.float32); hard[8:24, 8:24] = 1
        soft = feather(hard, 2.0)
        self.assertEqual(soft.shape, hard.shape)
        self.assertTrue(0 < soft[7, 15] < 1)
        self.assertGreater(float(soft[15, 15]), 0.98)
        self.assertTrue(np.array_equal(feather(hard, 0), hard))


class PipelineControlMaskScaleTests(unittest.TestCase):
    def test_control_mask_is_accepted_at_processing_scale_2(self):
        pipeline = NeuralRenderingPipeline(synthetic_weights(), device="cpu")
        rng = np.random.default_rng(0)
        image = rng.random((24, 32, 3), dtype=np.float32)
        result = pipeline.enhance(image, processing_scale=2, skin_mask=np.ones((24, 32), dtype=np.float32) * 0.5)
        self.assertEqual(result.image.shape, image.shape)


@unittest.skipUnless(os.environ.get("MLXDLSS_TEST_AUTOMASK"), "set MLXDLSS_TEST_AUTOMASK=1 to run the face-parsing model")
class SkinMaskerModelTests(unittest.TestCase):
    def test_masks_a_face(self):
        from PIL import Image

        image = np.asarray(Image.open(os.environ["MLXDLSS_TEST_AUTOMASK"]).convert("RGB")).astype(np.float32) / 255
        masker = SkinMasker(device="cpu")
        skin = masker.skin(image)
        self.assertEqual(skin.shape, image.shape[:2])
        self.assertGreater(float(skin.mean()), 0.02)
        self.assertLess(float(skin.mean()), 0.9)
        self.assertTrue(set(np.unique(masker.labels(image))) & set(SKIN_LABELS))
