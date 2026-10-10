import unittest
import os
import sys
import tempfile
from unittest.mock import MagicMock
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.sdcli.utils import (
    load_config,
    evaluate_fstring,
    read_file_as_fstring,
    embed_metadata_to_png,
)
from sdcli import apply_overrides


class TestUtils(unittest.TestCase):
    def test_evaluate_fstring_valid(self):
        ctx = {"character": "Motoko", "attire": "cybernetic suit", "mood": "focused"}
        prompt = "masterpiece, solo, {character}, wearing {attire}, {mood}"
        result = evaluate_fstring(prompt, ctx)
        self.assertEqual(
            result, "masterpiece, solo, Motoko, wearing cybernetic suit, focused"
        )

    def test_evaluate_fstring_with_expression(self):
        ctx = {"character": "Motoko", "seed": 42}
        prompt = "solo, {character}, seed: {seed * 2}"
        result = evaluate_fstring(prompt, ctx)
        self.assertEqual(result, "solo, Motoko, seed: 84")

    def test_evaluate_fstring_preserves_prompt_weights_and_unresolved(self):
        ctx = {"character": "Motoko"}
        # Should preserve {masterpiece:1.2} and {undefined_var} without crashing
        prompt = "solo, {character}, {masterpiece:1.2}, {undefined_var}"
        result = evaluate_fstring(prompt, ctx)
        self.assertEqual(
            result, "solo, Motoko, {masterpiece:1.2}, {undefined_var}"
        )

    def test_evaluate_fstring_blocks_dunder_traversal(self):
        ctx = {"character": "Motoko"}
        malicious = "{random.choice.__globals__}"
        result = evaluate_fstring(malicious, ctx)
        self.assertEqual(result, malicious)

    def test_apply_overrides_dest_path_expansion(self):
        config = {"image_path": "/tmp/original"}
        args = MagicMock()
        args.upscale = True
        args.refine = True
        args.dest = "~/my_artwork"

        apply_overrides(config, args)

        self.assertFalse(config["image_path"].startswith("~"))
        self.assertTrue(os.path.isabs(config["image_path"]))
        self.assertTrue(config["image_path"].endswith("my_artwork"))

    def test_read_file_as_fstring(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".prompt") as f:
            f.write("# Header comment\n")
            f.write("   # Indented comment\n")
            f.write("solo, {character}, in {place},\n")
            f.write("# Another comment\n")
            f.write("high quality\n")
            filepath = f.name

        try:
            ctx = {"character": "Motoko", "place": "Neo Tokyo"}
            result = read_file_as_fstring(filepath, ctx)
            self.assertIn("solo, Motoko, in Neo Tokyo,", result)
            self.assertIn("high quality", result)
            self.assertNotIn("comment", result)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_config_path_expansion(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as f:
            f.write("image_path: ~/Pictures/test_sd\n")
            f.write("video_path: ~/Videos/test_sd\n")
            f.write("wildcards_dir: ~/test_wildcards\n")
            filepath = f.name

        try:
            config = load_config(filepath)
            self.assertFalse(config["image_path"].startswith("~"))
            self.assertFalse(config["video_path"].startswith("~"))
            self.assertFalse(config["wildcards_dir"].startswith("~"))
            self.assertTrue(os.path.isabs(config["image_path"]))
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_embed_metadata_to_png(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".png") as f:
            filepath = f.name

        try:
            img = Image.new("RGB", (32, 32), color=(255, 0, 0))
            img.save(filepath)

            meta = {"Author": "Renich Bon Ćirić", "License": "CC-BY-SA 4.0"}
            embed_metadata_to_png(filepath, meta)

            with Image.open(filepath) as reloaded:
                self.assertEqual(reloaded.info.get("Author"), "Renich Bon Ćirić")
                self.assertEqual(reloaded.info.get("License"), "CC-BY-SA 4.0")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
