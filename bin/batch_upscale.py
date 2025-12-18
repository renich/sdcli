#!/home/renich/Projects/hf/sdcli/venv/bin/python

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import yaml
import numpy as np
from tqdm import tqdm
import torch
from spandrel import ModelLoader

def load_upscaler(config_path, device):
    """Load upscaler model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    upscaler_config = config.get('upscaling', {})
    model_path = upscaler_config.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        print(f"Upscaler model not found: {model_path}")
        return None
    
    try:
        loader = ModelLoader()
        model = loader.load_from_file(model_path)
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        print(f"Failed to load upscaler: {e}")
        return None

def upscale_image(image_path, output_dir, model, scale_factor=4):
    """Upscale a single image."""
    try:
        # Load image
        image = Image.open(image_path)
        width, height = image.size
        new_width, new_height = width * scale_factor, height * scale_factor
        
        print(f"Processing: {os.path.basename(image_path)} ({width}x{height} -> {new_width}x{new_height})")
        
        # Convert to tensor and process
        with torch.no_grad():
            img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
            output = model(img_tensor)
            output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        
        # Save upscaled image
        upscaled_image = Image.fromarray(output)
        
        # Generate output filename
        input_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{input_name}_upscaled_{scale_factor}x.png")
        
        upscaled_image.save(output_path)
        print(f"Saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def batch_upscale(input_dir, output_dir, config_path, pattern="*.png"):
    """Batch upscale all images in directory."""
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load upscaler
    model = load_upscaler(config_path, device)
    if model is None:
        return
    
    # Find and process images
    image_files = list(Path(input_dir).glob(pattern))
    if not image_files:
        print("No images found to upscale.")
        return
    
    print(f"Found {len(image_files)} images to upscale...")
    
    # Process each image
    successful = []
    for image_file in tqdm(image_files, desc="Upscaling images"):
        result = upscale_image(str(image_file), output_dir, model, scale_factor=4)
        if result:
            successful.append(result)
    
    print(f"\nSuccessfully upscaled {len(successful)}/{len(image_files)} images!")

def main():
    parser = argparse.ArgumentParser(description="Batch upscaler for existing 1024x1024 images")
    parser.add_argument("input_dir", help="Directory containing 1024x1024 images")
    parser.add_argument("output_dir", help="Directory to save upscaled images")
    parser.add_argument("--config", default="config.yaml", help="Config file for upscaler settings")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor (default: 4)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run batch upscaling
    batch_upscale(args.input_dir, args.output_dir, args.config)

if __name__ == "__main__":
    main()