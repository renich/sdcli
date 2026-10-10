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

    def test_get_target_prompt_and_negative_prompt(self):
        ref_cfg = {
            "prompt": "masterpiece",
            "negative_prompt": "ugly",
            "face_prompt": "custom face prompt",
            "world_negative_prompt": "custom world neg",
        }
        # Custom override
        self.assertEqual(
            self.generator._get_target_prompt(ref_cfg, "face", "orig"),
            "custom face prompt",
        )
        self.assertEqual(
            self.generator._get_target_negative_prompt(ref_cfg, "world", "orig_neg"),
            "custom world neg",
        )

        # Fallback with anchor tags
        body_prompt = self.generator._get_target_prompt(ref_cfg, "body", "orig")
        self.assertIn("masterpiece", body_prompt)
        self.assertIn("perfect anatomy", body_prompt)

        face_neg = self.generator._get_target_negative_prompt(
            ref_cfg, "face", "orig_neg"
        )
        self.assertIn("ugly", face_neg)
        self.assertIn("crooked face", face_neg)

    def test_detail_world(self):
        w, h = 64, 64
        test_img = Image.new("RGB", (w, h), color=(100, 100, 100))
        person_mask = np.zeros((h, w), dtype=bool)
        person_mask[20:40, 20:40] = True

        mock_out = Image.new("RGB", (w, h), color=(200, 200, 200))
        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[mock_out])
        self.generator.img2img_pipeline = mock_pipe
        self.generator.generator = None

        result = self.generator._detail_world(
            test_img,
            [person_mask],
            "orig_prompt",
            "orig_neg",
            {"world_strength": 0.25, "world_steps": 10},
        )
        self.assertEqual(result.size, (w, h))
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        self.assertEqual(kwargs["strength"], 0.25)
        self.assertEqual(kwargs["num_inference_steps"], 10)

    def test_detail_body(self):
        w, h = 64, 64
        test_img = Image.new("RGB", (w, h), color=(100, 100, 100))
        person_mask = np.zeros((h, w), dtype=bool)
        person_mask[20:40, 20:40] = True

        mock_out = Image.new("RGB", (w, h), color=(210, 210, 210))
        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[mock_out])
        self.generator.img2img_pipeline = mock_pipe
        self.generator.generator = None

        result = self.generator._detail_body(
            test_img,
            [person_mask],
            [[20, 20, 40, 40]],
            "orig_prompt",
            "orig_neg",
            {"body_strength": 0.2, "body_steps": 15},
        )
        self.assertEqual(result.size, (w, h))
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        self.assertEqual(kwargs["strength"], 0.2)
        self.assertEqual(kwargs["num_inference_steps"], 15)

    def test_detail_face(self):
        w, h = 128, 128
        test_img = Image.new("RGB", (w, h), color=(100, 100, 100))
        face_boxes = [[40, 40, 60, 60]]

        mock_crop_out = Image.new("RGB", (768, 768), color=(220, 220, 220))
        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[mock_crop_out])
        self.generator.img2img_pipeline = mock_pipe
        self.generator.generator = None

        result = self.generator._detail_face(
            test_img,
            face_boxes,
            "orig_prompt",
            "orig_neg",
            {"face_strength": 0.35, "face_steps": 20},
        )
        self.assertEqual(result.size, (w, h))
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        self.assertEqual(kwargs["strength"], 0.35)
        self.assertEqual(kwargs["image"].size, (768, 768))

    def test_refine_global_multi_target_flow(self):
        w, h = 64, 64
        test_img = Image.new("RGB", (w, h), color=(50, 50, 50))
        self.generator.config["refiner"] = {
            "enabled": True,
            "use_sam": True,
            "targets": ["world", "body", "face"],
        }
        self.generator.img2img_pipeline = MagicMock()

        person_mask = np.zeros((h, w), dtype=bool)
        person_mask[20:50, 20:50] = True
        self.generator._detect_faces = MagicMock(return_value=[[25, 25, 35, 35]])
        self.generator._detect_humans = MagicMock(return_value=[[20, 20, 50, 50]])
        self.generator._cached_person_masks = [person_mask]

        calls = []

        def mock_world(image, masks, p, np, cfg):
            calls.append("world")
            return image

        def mock_body(image, masks, boxes, p, np, cfg):
            calls.append("body")
            return image

        def mock_face(image, boxes, p, np, cfg):
            calls.append("face")
            return image

        self.generator._detail_world = mock_world
        self.generator._detail_body = mock_body
        self.generator._detail_face = mock_face

        result = self.generator._refine_global(test_img, "prompt", "neg_prompt")
        self.assertIsNotNone(result)
        # Verify strict execution order: world -> body -> face
        self.assertEqual(calls, ["world", "body", "face"])


if __name__ == "__main__":
    unittest.main()
