#!/home/renich/Projects/hf/sdcli/venv/bin/python

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from spandrel import ModelLoader


def load_upscaler(model_path):
    """Load upscaler model."""
    if not os.path.exists(model_path):
        print(f"Upscaler model not found: {model_path}")
        return None

    try:
        loader = ModelLoader()
        model = loader.load_from_file(model_path)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model
    except Exception as e:
        print(f"Failed to load upscaler: {e}")
        return None


def upscale_single_image(image_path, model, output_dir, scale_factor=4):
    """Upscale a single 1024x1024 image to 4096x4096."""
    try:
        # Load image
        image = Image.open(image_path)
        width, height = image.size
        new_width, new_height = width * scale_factor, height * scale_factor

        print(
            "Processing: "
            + os.path.basename(image_path)
            + " ("
            + str(width)
            + "x"
            + str(height)
            + " -> "
            + str(new_width)
            + "x"
            + str(new_height)
            + ")"
        )

        # Simple tensor upscaling
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            img_tensor = (
                torch.tensor(np.array(image))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255)
                .to(device)
            )
            output = model(img_tensor)
            output = (
                output.squeeze(0)
                .permute(1, 2, 0)
                .clamp(0, 1)
                .mul(255)
                .byte()
                .cpu()
                .numpy()
            )

        # Save upscaled image
        upscaled_image = Image.fromarray(output)

        # Generate output filename
        input_name = Path(image_path).stem
        output_path = os.path.join(
            output_dir, f"{input_name}_upscaled_{scale_factor}x.png"
        )

        upscaled_image.save(output_path)
        print(f"Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def batch_upscale(input_dir, output_dir, model_path, scale_factor=4):
    """Batch upscale all 1024x1024 images in directory."""
    # Find 1024x1024 images
    image_files = []
    for f in Path(input_dir).glob("*.png"):
        try:
            with Image.open(f) as img:
                if img.size == (1024, 1024):
                    image_files.append(f)
        except Exception:
            continue
    if not image_files:
        print("No 1024x1024 images found to upscale.")
        return

    print(f"Found {len(image_files)} 1024x1024 images to upscale...")

    # Load upscaler
    model = load_upscaler(model_path)
    if model is None:
        return

    # Process each image
    successful = []
    for image_file in tqdm(image_files, desc="Upscaling 1024x1024 images"):
        result = upscale_single_image(str(image_file), model, output_dir, scale_factor)
        if result:
            successful.append(result)

    print(f"\nSuccessfully upscaled {len(successful)}/{len(image_files)} images!")


def main():
    parser = argparse.ArgumentParser(
        description="Batch upscale 1024x1024 images to 4096x4096"
    )
    parser.add_argument("input_dir", help="Directory containing 1024x1024 images")
    parser.add_argument("output_dir", help="Directory to save upscaled images")
    parser.add_argument(
        "--model-path",
        default="/usr/local/share/models/comfyui/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth",
        help="Path to upscaler model",
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="Scale factor (default: 4)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir

    # Run batch upscaling
    batch_upscale(args.input_dir, output_dir, args.model_path, args.scale)


if __name__ == "__main__":
    main()
