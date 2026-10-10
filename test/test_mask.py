import unittest
import os
import sys
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.init import ImageGenerator


class TestMaskLogic(unittest.TestCase):
    def setUp(self):
        self.generator = ImageGenerator.__new__(ImageGenerator)
        self.generator.config = {
            "refiner": {
                "enabled": True,
                "use_sam": True,
                "targets": ["face", "body"],
                "debug_yolo": False,
            }
        }
        self.generator.sam_predictor = MagicMock()

    def test_generate_world_mask(self):
        h, w = 10, 10
        image_np = np.zeros((h, w, 3), dtype=np.uint8)

        # Create two mock person masks
        mask1 = np.zeros((h, w), dtype=bool)
        mask1[0:3, 0:3] = True
        mask2 = np.zeros((h, w), dtype=bool)
        mask2[2:5, 2:5] = True

        world_mask = self.generator._generate_world_mask(image_np, [mask1, mask2])

        self.assertEqual(world_mask.shape, (h, w))
        self.assertTrue(world_mask[9, 9])
        self.assertFalse(world_mask[0, 0])
        self.assertFalse(world_mask[3, 3])

    def test_generate_sam_mask_no_detections(self):
        # When no faces or humans are detected, it must return None (graceful skip)
        self.generator._detect_faces = MagicMock(return_value=[])
        self.generator._detect_humans = MagicMock(return_value=[])

        test_img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        result = self.generator._generate_sam_mask(test_img)

        self.assertIsNone(result)

    def test_generate_sam_mask_with_face_and_body(self):
        h, w = 32, 32
        test_img = Image.new("RGB", (w, h), color=(100, 100, 100))

        face_mask = np.zeros((h, w), dtype=bool)
        face_mask[2:10, 2:10] = True
        body_mask = np.zeros((h, w), dtype=bool)
        body_mask[8:25, 5:20] = True

        self.generator._detect_faces = MagicMock(return_value=[[2, 2, 10, 10]])
        self.generator._detect_humans = MagicMock(return_value=[[5, 8, 20, 25]])
        self.generator._refine_box_with_sam = MagicMock(
            side_effect=[(face_mask, 0.95), (body_mask, 0.88)]
        )

        result_mask = self.generator._generate_sam_mask(test_img)

        self.assertIsNotNone(result_mask)
        self.assertEqual(result_mask.size, (w, h))
        self.assertEqual(result_mask.mode, "L")

        mask_np = np.array(result_mask)
        unique_vals = set(np.unique(mask_np))
        self.assertTrue(unique_vals.issubset({0, 255}))
        # Face area is white (255)
        self.assertEqual(mask_np[4, 4], 255)
        # Background is black (0)
        self.assertEqual(mask_np[30, 30], 0)

    def test_generate_sam_mask_world_target(self):
        h, w = 32, 32
        test_img = Image.new("RGB", (w, h), color=(100, 100, 100))
        self.generator.config["refiner"]["targets"] = ["world"]

        person_mask = np.zeros((h, w), dtype=bool)
        person_mask[10:20, 10:20] = True

        self.generator._detect_faces = MagicMock(return_value=[])
        self.generator._detect_humans = MagicMock(return_value=[[10, 10, 20, 20]])
        self.generator._refine_box_with_sam = MagicMock(
            return_value=(person_mask, 0.9)
        )

        result_mask = self.generator._generate_sam_mask(test_img)

        self.assertIsNotNone(result_mask)
        mask_np = np.array(result_mask)
        # Person area must be 0 (preserved)
        self.assertEqual(mask_np[15, 15], 0)
        # World area must be 255 (inpainted)
        self.assertEqual(mask_np[0, 0], 255)


if __name__ == "__main__":
    unittest.main()
