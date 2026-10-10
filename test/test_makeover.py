import unittest
from unittest.mock import MagicMock
import os
import sys

# Set up imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.components import WildcardManager
from sdcli import apply_overrides


class TestMakeover(unittest.TestCase):
    def test_wildcard_manager_singular_plural(self):
        # Create a mock WildcardManager without calling __init__
        manager = WildcardManager.__new__(WildcardManager)
        manager.wildcards_dir = "mock_dir"

        # Mock self.cache
        manager.cache = {
            "characters": ["Jessie", "Cloud"],
            "action": ["running", "jumping"],
            "classes": ["warrior", "mage"],
            "glass": ["cup", "mug"],  # ends in ss
        }

        # Test get_context
        context = manager.get_context()

        # Assert both plural and singular exist for 'characters'
        self.assertIn("characters", context)
        self.assertIn("character", context)
        self.assertEqual(context["characters"], context["character"])

        # Assert both plural and singular exist for 'action'
        self.assertIn("action", context)
        self.assertIn("actions", context)
        self.assertEqual(context["action"], context["actions"])

        # Assert both plural and singular exist for 'classes' (ends in es, with s before it)
        self.assertIn("classes", context)
        self.assertIn("class", context)
        self.assertEqual(context["classes"], context["class"])

        # Assert 'glass' does not map to 'glas' (ends in ss)
        self.assertIn("glass", context)
        self.assertIn("glasses", context)
        self.assertNotIn("glas", context)
        self.assertEqual(context["glass"], context["glasses"])

    def test_apply_overrides(self):
        config = {
            "image_path": "original_path",
            "upscaling": {"enabled": True},
            "refiner": {"enabled": True},
        }

        # Mock args
        args = MagicMock()
        args.upscale = False
        args.refine = False
        args.dest = "new_dest_path"

        apply_overrides(config, args)

        self.assertEqual(config["image_path"], "new_dest_path")
        self.assertFalse(config["upscaling"]["enabled"])
        self.assertFalse(config["refiner"]["enabled"])


if __name__ == "__main__":
    unittest.main()
