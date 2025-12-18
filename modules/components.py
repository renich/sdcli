import os
import random
import logging
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from spandrel import ModelLoader

class WildcardManager:
    """Manages loading and sampling from wildcard text files."""
    def __init__(self, wildcards_dir: str):
        self.wildcards_dir = wildcards_dir
        self.cache = {}
        self._load_wildcards()

    def _load_wildcards(self):
        """Loads all .txt files from the wildcards directory into memory."""
        if not os.path.exists(self.wildcards_dir):
            logging.warning(f"Wildcards directory not found: {self.wildcards_dir}")
            return

        loaded_count = 0
        for filename in os.listdir(self.wildcards_dir):
            if filename.endswith(".txt"):
                key = os.path.splitext(filename)[0]
                path = os.path.join(self.wildcards_dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        self.cache[key] = lines
                        loaded_count += 1
                except Exception as e:
                    logging.warning(f"Failed to load wildcard {filename}: {e}")
        
        logging.info(f"Loaded {loaded_count} wildcard categories.")

    def get_context(self, iteration=None) -> dict:
        """Returns a dictionary with a random value for every loaded wildcard category."""
        context = {key: random.choice(values) for key, values in self.cache.items()}
        
        # Add iteration-specific context
        if iteration is not None:
            context['iteration'] = iteration
            context['iteration_total'] = iteration + 1
            
        return context


class Upscaler:
    """Handles image upscaling using Spandrel."""
    def __init__(self, config: dict, device: torch.device):
        self.config = config.get('upscaling', {})
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.config.get('enabled'):
            return

        path = self.config.get('model_path')
        if not path or not os.path.exists(path):
            logging.warning(f"Upscaler model not found at {path}")
            return

        logging.info(f"Loading Upscaler: {os.path.basename(path)}")
        try:
            loader = ModelLoader()
            self.model = loader.load_from_file(path)
            self.model.eval()
            self.model.to(self.device)
        except Exception as e:
            logging.error(f"Failed to load upscaler: {e}")

    def process(self, image: Image.Image) -> Image.Image:
        if self.model is None or not self.config.get('enabled'):
            return image

        logging.info("Step: Upscaling (Tiled)...")
        scale = self.config.get('scale_factor', 4)
        tile_size = self.config.get('tile_size', 512)
        overlap = self.config.get('tile_overlap', 32)

        width, height = image.size
        new_width, new_height = width * scale, height * scale
        upscaled_image = Image.new("RGB", (new_width, new_height))
        
        # Generate tiles
        tiles = []
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                tiles.append((x, y))

        for (x, y) in tqdm(tiles, desc="Upscaling Tiles", unit="tile"):
            # Crop
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            tile = image.crop(box)
            
            # Inference
            with torch.no_grad():
                img_tensor = torch.tensor(np.array(tile)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.device)
                output = self.model(img_tensor)
                output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            
            # Paste
            upscaled_tile = Image.fromarray(output)
            upscaled_image.paste(upscaled_tile, (x * scale, y * scale))

        return upscaled_image
