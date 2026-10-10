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
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Expand user paths
    if "image_path" in config:
        config["image_path"] = os.path.expanduser(config["image_path"])
    if "video_path" in config:
        config["video_path"] = os.path.expanduser(config["video_path"])
    if "wildcards_dir" in config:
        config["wildcards_dir"] = os.path.expanduser(config["wildcards_dir"])
    return config


def evaluate_fstring(text, context, source="<string>"):
    """Evaluates text substituting wildcards and safe expressions using the provided context."""
    if text is None:
        return None

    # Replace multiple spaces after commas with a single space
    cleaned_text = re.sub(r",\s+", ", ", text.strip())

    safe_globals = {
        "__builtins__": {},
        "random": random,
        "choice": random.choice,
        "randint": random.randint,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "len": len,
    }

    pattern = re.compile(r"\{([^{}]+)\}")

    def replacer(match):
        expr = match.group(1).strip()
        # 1. Direct wildcard key match
        if expr in context:
            return str(context[expr])

        # 2. Block prohibited dunder traversal
        if "__" in expr:
            logging.warning(
                f"Prohibited dunder access in prompt token from {source}: '{expr}'"
            )
            return match.group(0)

        # 3. Attempt safe expression evaluation (e.g. random.choice)
        try:
            val = eval(expr, safe_globals, context)
            return str(val)
        except Exception:
            # Preserve non-evaluable tokens (e.g. SD weighting syntax like {masterpiece:1.2})
            return match.group(0)

    return pattern.sub(replacer, cleaned_text)


def read_file_as_fstring(filename, context):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {filename}")
        return None
    except Exception as e:
        logging.error(f"Error reading prompt file {filename}: {e}")
        return None

    non_comment_lines = [line for line in lines if not line.strip().startswith("#")]
    fstring_text = "\n".join(non_comment_lines)
    return evaluate_fstring(fstring_text, context, source=filename)


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
