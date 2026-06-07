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
)
from modules.components import WildcardManager

console = Console()


def apply_overrides(config, args):
    if args.dest:
        config["video_path"] = args.dest
    if args.model_path is not None:
        config["video_model_path"] = args.model_path


def main():
    setup_logging()

    # CLI
    parser = argparse.ArgumentParser(
        description="Img2VidCLI - Image to Video Command Line Interface",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("image", type=str, help="Input image file path")
    parser.add_argument(
        "--prompt",
        type=str,
        default="positive.prompt",
        help="Prompt file or direct text",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="negative.prompt",
        help="Negative prompt file or direct text",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the video model directory (e.g. Wan diffusers). Pass empty string to fallback to Hugging Face models.",
    )
    parser.add_argument(
        "--dest", type=str, help="Destination directory for generated videos"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument("--guidance", type=float, default=9.0, help="Guidance scale")
    parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Number of frames to generate (Wan defaults to 81)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for output video (default 16)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Do not open the generated video in a player",
    )

    args = parser.parse_args()

    # Config
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Config Error: {e}")
        return

    apply_overrides(config, args)

    # Environment
    for k, v in config.get("environment", {}).items():
        os.environ[k] = str(v)

    if not os.path.exists(args.image):
        logging.error(f"Input image not found: {args.image}")
        return

    # Initialize Wildcards
    wildcards = WildcardManager(config.get("wildcards_dir", "wildcards"))

    # Execution Setup
    prompts_dir = config.get("prompts_dir", "prompts")
    ctx = wildcards.get_context()

    # Resolve prompts
    if os.path.exists(args.prompt):
        p_path = args.prompt
        logging.info(f"Using prompt path: {p_path} (resolved from CWD)")
        final_prompt = read_file_as_fstring(p_path, ctx)
    elif os.path.exists(os.path.join(prompts_dir, args.prompt)):
        p_path = os.path.join(prompts_dir, args.prompt)
        logging.info(f"Using prompt path: {p_path} (resolved from prompts_dir)")
        final_prompt = read_file_as_fstring(p_path, ctx)
    else:
        logging.info("Using direct text prompt")
        final_prompt = args.prompt

    if os.path.exists(args.negative_prompt):
        n_path = args.negative_prompt
        logging.info(f"Using negative prompt path: {n_path} (resolved from CWD)")
        final_nprompt = read_file_as_fstring(n_path, ctx) or ""
    elif os.path.exists(os.path.join(prompts_dir, args.negative_prompt)):
        n_path = os.path.join(prompts_dir, args.negative_prompt)
        logging.info(
            f"Using negative prompt path: {n_path} (resolved from prompts_dir)"
        )
        final_nprompt = read_file_as_fstring(n_path, ctx) or ""
    else:
        logging.info("Using direct text negative prompt")
        final_nprompt = args.negative_prompt

    if not final_prompt:
        logging.error("Empty prompt. Exiting.")
        return

    console.print(f"\n[bold white on green]Prompt:[/]\n{final_prompt}\n")
    console.print(f"[bold white on red]Negative:[/]\n{final_nprompt}\n")

    if args.dry_run:
        return

    # Initialize diffusers components
    model_path = config.get(
        "video_model_path", "/usr/local/share/models/wan2.2-diffusers"
    )
    use_wan = False
    if os.path.exists(model_path):
        import json

        idx_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(idx_path):
            try:
                with open(idx_path, "r") as f:
                    meta = json.load(f)
                if "WanImageToVideoPipeline" in meta.get("_class_name", ""):
                    use_wan = True
            except Exception:
                pass

    if use_wan:
        logging.info(f"Loading local WanImageToVideoPipeline from: {model_path}")
        try:
            import torch
            from diffusers import WanImageToVideoPipeline
            from diffusers.utils import export_to_video, load_image
            import numpy as np

            pipeline = WanImageToVideoPipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
            pipeline.enable_model_cpu_offload()
        except Exception as e:
            logging.error(f"Failed to load Wan pipeline: {e}")
            return
    else:
        logging.info("Loading I2VGenXLPipeline component from huggingface...")
        try:
            import torch
            from diffusers import I2VGenXLPipeline
            from diffusers.utils import export_to_video, load_image

            pipeline = I2VGenXLPipeline.from_pretrained(
                "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
            )
            pipeline.enable_model_cpu_offload()
        except Exception as e:
            logging.error(f"Failed to load video pipeline: {e}")
            return

    # Load image
    try:
        image = load_image(args.image).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to load image {args.image}: {e}")
        return

    # Generate
    logging.info("Generating video frames...")
    try:
        generator = torch.Generator(device="cuda")
        if use_wan:
            # Wan2.2 resolution calculations from README
            max_area = 480 * 832
            aspect_ratio = image.height / image.width
            mod_value = (
                pipeline.vae_scale_factor_spatial
                * pipeline.transformer.config.patch_size[1]
            )
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            image = image.resize((width, height))

            frames = pipeline(
                image=image,
                prompt=final_prompt,
                negative_prompt=final_nprompt,
                height=height,
                width=width,
                num_frames=args.frames,
                guidance_scale=3.5,
                num_inference_steps=40,
                generator=generator,
            ).frames[0]
        else:
            frames = pipeline(
                prompt=final_prompt,
                image=image,
                num_inference_steps=args.steps,
                negative_prompt=final_nprompt,
                guidance_scale=args.guidance,
                generator=generator,
            ).frames[0]
    except Exception as e:
        logging.error(f"Video generation failed: {e}")
        return

    # Save Setup
    save_base = os.path.expanduser(config.get("video_path", "~/Videos/artwork"))
    date_str = time.strftime("%Y-%m-%d")
    time_str = time.strftime("%H%M%S")
    full_date = time.strftime("%Y%m%d")

    out_dir = os.path.join(save_base, date_str)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{full_date}{time_str}.mp4"
    filepath = os.path.join(out_dir, filename)

    try:
        export_to_video(frames, filepath, fps=args.fps)
        logging.info(f"Saved video to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to export video: {e}")
        return

    # Open viewer
    if not args.no_view:
        viewer_cmd = config.get("video_viewer")
        if viewer_cmd:
            try:
                cmd = shlex.split(viewer_cmd) + [filepath]
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception as e:
                logging.warning(f"Failed to launch custom video viewer: {e}")
        else:
            import shutil

            # Fallback list of standard linux video players
            for player in ["mpv", "vlc", "totem", "celluloid", "xdg-open"]:
                player_bin = player.split()[0]
                if shutil.which(player_bin):
                    try:
                        cmd = shlex.split(player) + [filepath]
                        subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        break
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
