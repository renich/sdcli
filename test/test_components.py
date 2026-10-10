import unittest
import os
import sys
import tempfile
import torch
from unittest.mock import MagicMock
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.components import WildcardManager, Upscaler
from sdcli import resolve_prompt


class TestComponents(unittest.TestCase):
    def test_wildcard_manager_ignores_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "moods.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# This is a comment\n")
                f.write("cheerful\n")
                f.write("   # Indented comment\n")
                f.write("melancholic\n")
                f.write("\n")

            manager = WildcardManager(tmpdir)
            self.assertIn("moods", manager.cache)
            self.assertEqual(manager.cache["moods"], ["cheerful", "melancholic"])

            ctx = manager.get_context()
            self.assertIn(ctx["mood"], ["cheerful", "melancholic"])
            self.assertEqual(ctx["mood"], ctx["moods"])

    def test_upscaler_rescaling(self):
        config = {
            "upscaling": {
                "enabled": True,
                "scale_factor": 2,
            }
        }
        device = torch.device("cpu")
        upscaler = Upscaler.__new__(Upscaler)
        upscaler.config = config["upscaling"]
        upscaler.device = device

        # Mock model that doubles image size twice (4x output)
        mock_model = MagicMock()

        def fake_model(tensor):
            # simulate 4x scale: tensor shape (1, C, H, W) -> (1, C, H*4, W*4)
            _, c, h, w = tensor.shape
            return torch.zeros((1, c, h * 4, w * 4), dtype=torch.float32)

        mock_model.side_effect = fake_model
        upscaler.model = mock_model

        input_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = upscaler.process(input_img)

        # Since scale_factor is 2, and model output 4x (256x256), Upscaler should rescale to 128x128 (2x)
        self.assertEqual(result.size, (128, 128))

    def test_upscaler_rgba_and_grayscale(self):
        config = {
            "upscaling": {
                "enabled": True,
                "scale_factor": 4,
            }
        }
        device = torch.device("cpu")
        upscaler = Upscaler.__new__(Upscaler)
        upscaler.config = config["upscaling"]
        upscaler.device = device

        mock_model = MagicMock()

        def fake_model(tensor):
            # Assert tensor always has 3 channels regardless of input mode
            _, c, h, w = tensor.shape
            self.assertEqual(c, 3)
            return torch.zeros((1, 3, h * 4, w * 4), dtype=torch.float32)

        mock_model.side_effect = fake_model
        upscaler.model = mock_model

        # Test RGBA image
        rgba_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
        res_rgba = upscaler.process(rgba_img)
        self.assertEqual(res_rgba.size, (128, 128))

        # Test Grayscale (L) image
        gray_img = Image.new("L", (32, 32), color=128)
        res_gray = upscaler.process(gray_img)
        self.assertEqual(res_gray.size, (128, 128))

    def test_resolve_prompt_direct_text(self):
        ctx = {"character": "Motoko", "action": "jumping"}
        direct_text = "masterpiece, {character}, {action}"
        result = resolve_prompt(direct_text, prompts_dir="/nonexistent_dir", ctx=ctx)
        self.assertEqual(result, "masterpiece, Motoko, jumping")


if __name__ == "__main__":
    unittest.main()
