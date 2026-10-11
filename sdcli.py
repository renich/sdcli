#! ./venv/bin/python

import os
import time
import pathlib
import argparse
import logging
import shlex
import subprocess
from rich.console import Console
from rich_argparse import RichHelpFormatter

from PIL.PngImagePlugin import PngInfo
from modules.sdcli.utils import (
    load_config,
    setup_logging,
    read_file_as_fstring,
    evaluate_fstring,
    get_output_filepath,
)
from modules.components import WildcardManager

console = Console()


def resolve_prompt(prompt_input, prompts_dir, ctx):
    """Resolves prompt input from CWD file, prompts directory, or direct text."""
    if not prompt_input:
        return ""

    if os.path.exists(prompt_input):
        logging.info(f"Using prompt path: {prompt_input} (resolved from CWD)")
        return read_file_as_fstring(prompt_input, ctx)

    p_path = os.path.join(prompts_dir, prompt_input)
    if os.path.exists(p_path):
        logging.info(f"Using prompt path: {p_path} (resolved from prompts_dir)")
        return read_file_as_fstring(p_path, ctx)

    logging.info("Using direct text prompt")
    return evaluate_fstring(prompt_input, ctx, source="CLI prompt")


def apply_overrides(config, args):
    if "upscaling" not in config:
        config["upscaling"] = {}
    if "refiner" not in config:
        config["refiner"] = {}

    if not args.upscale:
        config["upscaling"]["enabled"] = False
    if not args.refine:
        config["refiner"]["enabled"] = False

    if args.dest:
        config["image_path"] = os.path.expanduser(args.dest)


def main():
    setup_logging()

    # CLI
    parser = argparse.ArgumentParser(
        description="SDCLI - Stable Diffusion Command Line Interface",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "pos_count",
        nargs="?",
        type=int,
        default=None,
        help="Number of prompts to process (positional shorthand)",
    )
    parser.add_argument(
        "pos_batch_size",
        nargs="?",
        type=int,
        default=None,
        help="Number of images per prompt in parallel (positional shorthand)",
    )
    parser.add_argument("--count", type=int, help="Number of prompts to process")
    parser.add_argument(
        "--batch-size", type=int, help="Number of images per prompt (in parallel)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="positive.prompt",
        help="Prompt file or raw text",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="negative.prompt",
        help="Negative prompt file or raw text",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument(
        "--dest", type=str, help="Destination directory for generated images"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Exact output filepath (overrides dest, disables batch indexing)",
    )

    # Flags (defaults true in config, these allow disabling)
    parser.add_argument("--no-upscale", action="store_false", dest="upscale")
    parser.add_argument("--no-refine", action="store_false", dest="refine")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Do not open the generated image in a viewer",
    )

    parser.set_defaults(upscale=True, refine=True)
    args = parser.parse_args()

    # Config
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Config Error: {e}")
        return

    # Overrides
    apply_overrides(config, args)

    # Environment
    for k, v in config.get("environment", {}).items():
        os.environ[k] = str(v)

    # Initialize Components
    from modules.init import ImageGenerator

    wildcards = WildcardManager(config.get("wildcards_dir", "wildcards"))
    generator = ImageGenerator(config)

    if not args.dry_run:
        try:
            generator.load_resources()
        except Exception as e:
            logging.error(f"Initialization Failed: {e}")
            return

    # Execution Loop
    count_arg = args.count if args.count is not None else args.pos_count
    batch_size_arg = (
        args.batch_size if args.batch_size is not None else args.pos_batch_size
    )
    count = count_arg if count_arg is not None else config["generation"].get("count", 1)
    batch_size = (
        batch_size_arg
        if batch_size_arg is not None
        else config["generation"].get("batch_size", 1)
    )
    prompts_dir = config.get("prompts_dir", "prompts")

    logging.info(f"Starting: {count} iterations, {batch_size} images per iteration...")

    try:
        for i in range(count):
            logging.info(f"--- Iteration {i + 1}/{count} ---")

            # Reload config for each iteration (for dynamic adjustments)
            try:
                config = load_config(args.config)
                # Apply overrides
                apply_overrides(config, args)
                prompts_dir = config.get("prompts_dir", "prompts")

                # Update generator config if it exists
                if hasattr(generator, "config"):
                    generator.config.update(config)
            except Exception as e:
                logging.warning(f"Config reload failed for iteration {i + 1}: {e}")

            # 1. Prepare Prompts
            ctx = wildcards.get_context(iteration=i)

            final_prompt = resolve_prompt(args.prompt, prompts_dir, ctx)
            final_nprompt = resolve_prompt(args.negative_prompt, prompts_dir, ctx) or ""

            if not final_prompt:
                logging.error("Empty prompt. Skipping.")
                continue

            console.print(f"\n[bold white on green]Prompt:[/]\n{final_prompt}\n")
            console.print(f"[bold white on red]Negative:[/]\n{final_nprompt}\n")

            if args.dry_run:
                continue

            # 2. Generate Batch
            try:
                images, seed = generator.generate(
                    prompt=final_prompt, negative_prompt=final_nprompt, count=batch_size
                )
            except Exception as e:
                logging.error(f"Generation Failed: {e}")
                continue

            # 3. Process each image in batch
            for idx, base_image in enumerate(images):
                logging.info(f"Processing variation {idx + 1}/{batch_size}...")

                final_image = generator.process_image(
                    base_image, final_prompt, final_nprompt
                )

                # 4. Save
                if args.out:
                    out_path = pathlib.Path(args.out)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if count > 1 or batch_size > 1:
                        filepath = str(
                            out_path.parent
                            / f"{out_path.stem}_i{i + 1}_b{idx + 1}{out_path.suffix}"
                        )
                    else:
                        filepath = str(out_path)
                else:
                    filepath = get_output_filepath(
                        save_base=config["image_path"],
                        batch_idx=idx,
                        batch_size=batch_size,
                    )

                # Metadata in single save pass
                ops = []
                if config["upscaling"].get("enabled"):
                    ops.append("Upscaled")
                if config["refiner"].get("enabled"):
                    ops.append("Refined")
                ops_str = f" ({', '.join(ops)})" if ops else ""

                meta = {
                    "Author": "Renich Bon Ćirić",
                    "License": "CC-BY-SA 4.0",
                    "Parameters": f"{final_prompt} | Negative: {final_nprompt} | Steps: {config['generation']['steps']} | Seed: {seed} | Batch: {idx + 1}/{batch_size}{ops_str}",
                }
                png_info = PngInfo()
                for key, value in meta.items():
                    png_info.add_text(str(key), str(value))

                final_image.save(filepath, pnginfo=png_info)
                logging.info(f"Saved: {filepath}")

                # Viewer (only on first iteration and first image to prevent window spam)
                if i == 0 and idx == 0 and not args.no_view:
                    viewer_cmd = config.get("viewer")
                    if viewer_cmd:
                        try:
                            cmd = shlex.split(viewer_cmd) + [filepath]
                            subprocess.Popen(
                                cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logging.warning(f"Failed to launch custom viewer: {e}")
                    else:
                        import shutil

                        # Fallback list of standard linux image viewers
                        for viewer in ["eog", "xdg-open", "gio open"]:
                            viewer_bin = viewer.split()[0]
                            if shutil.which(viewer_bin):
                                try:
                                    cmd = shlex.split(viewer) + [filepath]
                                    subprocess.Popen(
                                        cmd,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                    )
                                    break
                                except Exception:
                                    pass

            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]Generation interrupted by user. Exiting gracefully...[/]"
        )
    finally:
        generator.free_memory()


if __name__ == "__main__":
    main()
