from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import re
import pathlib
import time
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


def get_output_dir(save_base, timestamp=None):
    """
    Returns the output directory partitioned by year and full date (YYYY/YYYY-MM-DD).
    Creates the directory if it does not exist.
    """
    if timestamp is None:
        timestamp = time.localtime()
    year_str = time.strftime("%Y", timestamp)
    date_str = time.strftime("%Y-%m-%d", timestamp)
    base = os.path.expanduser(save_base)
    out_dir = os.path.join(base, year_str, date_str)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def sanitize_character_name(name: str) -> str:
    """Converts character name into lowercase snake_case for filenames."""
    if not name:
        return ""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def extract_character_from_prompt(raw_prompt: str, context: dict) -> str | None:
    """
    If {character} or {characters} is referenced in raw_prompt, returns the evaluated character name.
    Otherwise returns None.
    """
    if not raw_prompt or not context:
        return None
    if re.search(r"\{[^{}]*\bcharacters?\b[^{}]*\}", raw_prompt):
        char = context.get("character") or context.get("characters")
        if char:
            return str(char)
    return None


def extract_character_from_image_metadata(
    image_path: str,
    wildcard_file: str = "/usr/local/share/models/comfyui/wildcards/character.txt",
) -> str | None:
    """
    Reads PNG image metadata and identifies the character name from the prompt using wildcard definitions.
    """
    try:
        with Image.open(image_path) as img:
            params = img.info.get("Parameters", "")
    except Exception as e:
        logging.warning(f"Failed to read image {image_path}: {e}")
        return None

    if not params:
        return None

    # Load wildcard list if available
    wildcard_chars = []
    if os.path.isfile(wildcard_file):
        try:
            with open(wildcard_file, "r", encoding="utf-8") as f:
                wildcard_chars = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            wildcard_chars.sort(key=len, reverse=True)
        except Exception as e:
            logging.warning(
                f"Failed to load wildcard characters from {wildcard_file}: {e}"
            )

    prompt_part = (
        params.split(" | Negative:")[0] if " | Negative:" in params else params
    )

    # 1. Match against known wildcard characters
    for char in wildcard_chars:
        pattern = r"\(" + re.escape(char) + r"(?::[0-9.]*)?\)"
        if re.search(pattern, prompt_part):
            return char
        if (
            f"({char})" in prompt_part
            or f" {char}," in prompt_part
            or f", {char} " in prompt_part
        ):
            return char

    # 2. Heuristic fallback: check for `(mini-giantess:...) (Character Name:weight)`
    match = re.search(
        r"\((?:mini-giantess:[^)]+\)\s*\()([^:)]+)(?::[0-9.]*)?\)", prompt_part
    )
    if match:
        return match.group(1).strip()

    return None


def get_output_filepath(
    save_base,
    timestamp=None,
    batch_idx=0,
    batch_size=1,
    extension=".png",
    character=None,
):
    """
    Constructs the target filepath within save_base/YYYY/YYYY-MM-DD/YYYYMMDDHHMMSS[-character][_idx].png.
    """
    if timestamp is None:
        timestamp = time.localtime()
    out_dir = get_output_dir(save_base, timestamp)
    time_str = time.strftime("%H%M%S", timestamp)
    full_date = time.strftime("%Y%m%d", timestamp)

    char_slug = sanitize_character_name(character) if character else ""
    char_part = f"-{char_slug}" if char_slug else ""
    suffix = f"_{batch_idx}" if batch_size > 1 else ""
    filename = f"{full_date}{time_str}{char_part}{suffix}{extension}"
    return os.path.join(out_dir, filename)


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
