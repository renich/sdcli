=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
============

[0.3.0] - 2026-10-11
--------------------

.. rubric:: Added

* Sequential multi-target detailer executing refinement across world, body, and face.
* High-resolution crop-based face inpainting with padding, edge feathering, and bounding box detection via YOLO12 (``yolo12x-face.pt``).
* Ultralytics YOLOv8m human segmentation (``person_yolov8m-seg.pt``) for body refinement and inverted dilated background inpainting.
* Dynamic character slug extraction from prompt templates, saving filenames as ``YYYYMMDDHHmmss-character_name.ext`` when ``{character}`` is present.
* Hierarchical date-based storage routing images to ``<save_base>/YYYY/<YYYY-MM-DD>/``.
* Automated metadata extraction to backfill legacy filenames from embedded PNG parameter chunks.
* Video generation CLI ``img2vidcli.py`` powered by Wan2.2 diffusers pipeline.
* Batch 4x upscaler utility ``bin/batch_upscale_4x.py`` using Spandrel ESRGAN.
* Comprehensive unit test suite covering prompt evaluation, metadata embedding, directory routing, and crop detailer operations.

.. rubric:: Changed

* Refactored pipeline initialization to share UNet and VAE weights across generation and inpainting stages via ``from_pipe``, eliminating redundant VRAM overhead.
* Modernized default upscaling factor to 4x superscale, outputting 4096x4096px masters.
* Replaced arbitrary f-string execution with AST-verified safe prompt interpolation blocking dunder traversals.
* Upgraded tokenizer handling to support arbitrary prompt lengths through weighted embedding chunking without truncation or length errors.

.. rubric:: Fixed

* Resolved diffusers ``torch_dtype`` deprecation by passing ``dtype`` explicitly during pipeline instantiation and offload.
* Eliminated false-positive FP32 casting warnings on ROCm/CUDA device transfers.
* Resolved Hugging Face Hub cache resolution warning by using direct snapshot references.
* Fixed YOLO confidence tensor logging formatting.
* Fixed garbage collection order to release Python objects prior to clearing PyTorch CUDA/ROCm cache.

[0.2.0] - 2025-12-18
--------------------

.. rubric:: Added

* Segment Anything Model 2 (SAM2) integration for targeted image segmentation.
* Global img2img refinement mode for artifact correction.
* Whole-image processing support in the Spandrel upscaler.
* Positional command-line arguments for prompt count and parallel batch size.

.. rubric:: Changed

* Optimized upscaler memory layout and simplified tiling pipeline.
* Reorganized configuration schema for modular generation and refiner settings.

[0.1.0] - 2024-09-29
--------------------

.. rubric:: Added

* Initial Stable Diffusion XL (SDXL) command-line interface.
* AUTOMATIC1111 style prompt weighting support.
* Independent VAE and scheduler configuration support.
* Dynamic prompt wildcards with python expression interpolation.
* Dual hardware acceleration support for AMD ROCm and NVIDIA CUDA.
