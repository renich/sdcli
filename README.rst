=====
sdcli
=====

Command-line generative image and video pipeline orchestrating Stable Diffusion XL, multi-target detailers, Spandrel upscaling, and Wan2.2.

Description
===========

sdcli is a high-performance command-line interface for local generative AI production. It executes text-to-image synthesis using Stable Diffusion XL (SDXL), conducts sequential multi-target crop refinement across world, body, and face regions, upscales outputs to 4096x4096px via Spandrel ESRGAN, and provides image-to-video generation using Wan2.2.

Features
========

* **SDXL Base Generation**: Native 1024x1024 synthesis with prompt weighting and customizable schedulers.
* **Shared Memory Architecture**: Generation and inpainting pipelines share base UNet and VAE weights via ``from_pipe``, preventing duplicate VRAM allocation.
* **Sequential Multi-Target Refinement**:
   * **World Detailing**: Background inpainting utilizing dilated inverse human segmentation masks.
   * **Body Detailing**: Subject inpainting guided by Ultralytics YOLOv8m human segmentation (``person_yolov8m-seg.pt``).
   * **Face Crop Detailing**: High-resolution face crop extraction via YOLO12 (``yolo12x-face.pt``), inpainting with edge feathering and seamless canvas insertion.
* **4K AI Upscaling**: Integrated Spandrel ESRGAN upscaling delivering 4096x4096px masters with full PNG metadata retention.
* **Structured Output Hierarchy**: Automatically organizes renders into date-partitioned storage: ``<save_base>/YYYY/<YYYY-MM-DD>/``.
* **Dynamic Character Filenames**: Automatically detects ``{character}`` in raw prompt templates and appends a lowercase snake_case slug: ``YYYYMMDDHHmmss-character_name.ext``.
* **AST-Safe Wildcard Evaluation**: Evaluates Python expressions and nested wildcards safely while blocking dangerous dunder traversals.
* **Image to Video Generation**: ``img2vidcli.py`` utility animating source images using Wan2.2 diffusion models.
* **Batch Upscaling**: Standalone utility ``bin/batch_upscale_4x.py`` for batch processing existing image archives.

Hardware Requirements
=====================

* **Operating System**: Linux (tested on Fedora Linux).
* **GPU**: AMD ROCm (tested on Radeon RX 7900 XTX / RDNA3) or NVIDIA CUDA (Compute Capability 8.0+).
* **VRAM**: 16 GB or higher recommended for native 1024x1024 generation, multi-target detailing, and 4x upscaling. Sequential CPU offloading can be toggled in ``config.yaml`` for constrained hardware.
* **Python**: Python 3.10+ (Python 3.12 recommended).

Installation
============

AMD ROCm Setup
--------------

.. code-block:: bash

   # create a virtual environment
   python3.12 -m venv venv
   source venv/bin/activate

   # install ROCm dependencies
   pip install --upgrade pip
   pip install --upgrade -r requirements-amd.txt

NVIDIA CUDA Setup
-----------------

.. code-block:: bash

   # create a virtual environment
   python3.12 -m venv venv
   source venv/bin/activate

   # install CUDA dependencies
   pip install --upgrade pip
   pip install --upgrade -r requirements-nvidia.txt

Configuration
=============

System settings and pipeline options are configured in ``config.yaml``:

.. code-block:: yaml

   image_path: ~/Pictures/artwork/text
   model_path: /usr/local/share/models/comfyui/checkpoints/base_model.safetensors
   vae_path: madebyollin/sdxl-vae-fp16-fix
   scheduler_path: stabilityai/stable-diffusion-xl-base-1.0
   wildcards_dir: /usr/local/share/models/comfyui/wildcards
   prompts_dir: prompts

   generation:
     width: 1024
     height: 1024
     steps: 50
     guidance_scale: 4
     scheduler: "EulerAncestral"
     enable_cpu_offload: false

   upscaling:
     enabled: true
     model_path: /usr/local/share/models/comfyui/upscale_models/4x_NMKD-Superscale-SP_178000_G.pth
     scale_factor: 4

   refiner:
     enabled: true
     targets: ["world", "body", "face"]
     yolo_face_model: /usr/local/share/models/comfyui/ultralytics/bbox/yolov12l-face.pt
     yolo_human_model: /usr/local/share/models/comfyui/ultralytics/segm/person_yolov8m-seg.pt
     face_crop_size: 768
     face_strength: 0.35
     face_steps: 25

Usage
=====

Basic Image Generation
----------------------

.. code-block:: bash

   # generate an image using prompts/default.prompt
   ./sdcli.py

   # generate 10 images sequentially (positional shorthand)
   ./sdcli.py 10

   # generate 10 images in 2 batches of 5 in parallel
   ./sdcli.py 2 5

   # generate from direct prompt text
   ./sdcli.py --prompt "cyberpunk city, neon lighting" --count 3

   # specify exact output filepath
   ./sdcli.py --out output/custom_render.png

Multi-Target Refinement Options
-------------------------------

.. code-block:: bash

   # bypass the refiner pass (base generation and upscale only)
   ./sdcli.py --no-refine

   # bypass the upscaler pass (base generation and refiner only)
   ./sdcli.py --no-upscale

   # suppress launching the desktop image viewer
   ./sdcli.py --no-view

Image to Video Generation
-------------------------

Animate static images into video clips using ``img2vidcli.py``:

.. code-block:: bash

   # generate video with custom motion prompt
   ./img2vidcli.py input.png --prompt "slow cinematic camera pan" --seed 42

   # configure frame count and playback framerate
   ./img2vidcli.py input.png --frames 81 --fps 24 --guidance 6.0

Batch Upscaling Utility
-----------------------

Upscale entire folders of 1024x1024 images to 4096x4096px using Spandrel:

.. code-block:: bash

   ./bin/batch_upscale_4x.py /path/to/source_dir /path/to/output_dir --scale 4

Prompts and Wildcards
=====================

Prompt Weighting
----------------

Prompts support AUTOMATIC1111 style weighting syntax:

.. code-block:: text

   masterpiece, (highly detailed eyes:1.4), [blurry background]

Wildcard Syntax and Evaluation
------------------------------

Wildcards reside in ``wildcards/`` (or the path defined in ``config.yaml``). Reference wildcards using f-string syntax:

.. code-block:: text

   solo, ({character}:1.4), wearing {clothing}, standing in {place}

You can embed Python expressions such as random selections:

.. code-block:: text

   {random.choice(['daylight', 'golden hour', 'night scenery'])}

Character Output Filenames
--------------------------

When ``{character}`` or ``{characters}`` is present in the prompt template, ``sdcli`` extracts the chosen character and saves the output as:

.. code-block:: text

   YYYYMMDDHHmmss-character_name_in_snake_case.png

When no character wildcard is used, filenames retain the standard date timestamp format:

.. code-block:: text

   YYYYMMDDHHmmss.png

Directory Layout
================

Generated images are stored hierarchically:

.. code-block:: text

   ~/Pictures/artwork/text/
   ├── 2026/
   │   ├── 2026-10-10/
   │   │   ├── 20261010161549-female_dancer.png
   │   │   └── 20261010164957-mina_ashiro_kaiju_no_8.png
   │   └── 2026-10-11/
   │       ├── 20261011001121-iris_final_fantasy_xv.png
   │       └── 20261011011003-li_mei_mortal_kombat.png
