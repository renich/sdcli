from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import re
import yaml
import os
import logging
from rich.logging import RichHandler


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Expand user paths
    config["image_path"] = os.path.expanduser(config["image_path"])
    return config


def read_file_as_fstring(filename, context):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {filename}")
        return None

    non_comment_lines = [line for line in lines if not line.startswith("#")]

    # Join the remaining lines into a single string with newlines
    fstring_text = "\n".join(non_comment_lines)

    # Replace multiple spaces after commas with a single space
    fstring_text = re.sub(r",\s+", ", ", fstring_text)

    # Safe evaluation context
    # We restrict globals to essential modules if needed, or empty to prevent arbitrary code execution
    # But since the user likes the flexibility, we'll allow random.
    safe_globals = {
        "__builtins__": {},
        "random": random,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "len": len,
    }

    try:
        # evaluate the f-string wrapper
        # We explicitly wrap the content in f"""...""" so the user file is treated as the *inside* of an f-string
        expression = f'f"""{fstring_text}"""'
        result = eval(expression, safe_globals, context)
    except Exception as e:
        logging.error(f"Error evaluating template {filename}: {e}")
        return None

    return result


def embed_metadata_to_png(image_path, metadata):
    """
    Embeds metadata into a PNG image.

    Args:
      image_path: The path to the PNG image file.
      metadata: A dictionary containing the metadata to embed.
    """
    try:
        img = Image.open(image_path)
        png_info = PngInfo()

        for key, value in metadata.items():
            png_info.add_text(str(key), str(value))

        img.save(image_path, pnginfo=png_info)
    except Exception as e:
        logging.error(f"Failed to embed metadata: {e}")
