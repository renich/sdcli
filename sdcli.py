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

# Utils
from modules.sdcli.utils import (
    load_config,
    setup_logging,
    read_file_as_fstring,
    embed_metadata_to_png,
)
from modules.components import WildcardManager

console = Console()


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
        config["image_path"] = args.dest


def main():
    setup_logging()

    # CLI
    parser = argparse.ArgumentParser(
        description="SDCLI - Stable Diffusion Command Line Interface",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("--count", type=int, help="Number of prompts to process")
    parser.add_argument(
        "--batch-size", type=int, help="Number of images per prompt (in parallel)"
    )
    parser.add_argument(
        "--prompt", type=str, default="positive.prompt", help="Prompt file"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="negative.prompt",
        help="Negative prompt file",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument(
        "--dest", type=str, help="Destination directory for generated images"
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
    count = (
        args.count if args.count is not None else config["generation"].get("count", 1)
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
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

            if os.path.exists(args.prompt):
                p_path = args.prompt
                logging.info(f"Using prompt path: {p_path} (resolved from CWD)")
            else:
                p_path = os.path.join(prompts_dir, args.prompt)
                logging.info(f"Using prompt path: {p_path} (resolved from prompts_dir)")

            if os.path.exists(args.negative_prompt):
                n_path = args.negative_prompt
                logging.info(
                    f"Using negative prompt path: {n_path} (resolved from CWD)"
                )
            else:
                n_path = os.path.join(prompts_dir, args.negative_prompt)
                logging.info(
                    f"Using negative prompt path: {n_path} (resolved from prompts_dir)"
                )

            final_prompt = read_file_as_fstring(p_path, ctx)
            final_nprompt = read_file_as_fstring(n_path, ctx) or ""

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
                save_base = config["image_path"]
                date_str = time.strftime("%Y-%m-%d")
                time_str = time.strftime("%H%M%S")
                full_date = time.strftime("%Y%m%d")

                out_dir = os.path.join(save_base, date_str)
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

                # If batch_size > 1, add index to filename
                suffix = f"_{idx}" if batch_size > 1 else ""
                filename = f"{full_date}{time_str}{suffix}.png"
                filepath = os.path.join(out_dir, filename)

                final_image.save(filepath)
                logging.info(f"Saved: {filepath}")

                # Metadata
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
                embed_metadata_to_png(filepath, meta)

                # Viewer (only for first one in batch to avoid spam?)
                if idx == 0 and not args.no_view:
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
