import gc
import logging
import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from diffusers import (
    StableDiffusionXLPAGPipeline,
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoPipelineForImage2Image,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
from modules.components import Upscaler

# YOLO imports for detection
from ultralytics import YOLO


class ImageGenerator:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.dtype = torch.float16
        self.generator = torch.Generator(self.device)

        # Pipelines
        self.pipeline = None
        self.img2img_pipeline = None

        # SAM
        self.sam_predictor = None

        # YOLO
        self.face_detector = None
        self.human_detector = None

        # Components
        self.upscaler = None

    def _get_device(self):
        if torch.cuda.is_available():
            if torch.version.hip:
                logging.info("Device: AMD GPU (ROCm)")
            else:
                logging.info("Device: NVIDIA GPU (CUDA)")
            return torch.device("cuda")
        return torch.device("cpu")

    def load_resources(self):
        """Loads all necessary models and pipelines."""
        self._load_pipelines()
        self._load_yolo_models()
        self.upscaler = Upscaler(self.config, self.device)

    def _load_pipelines(self):
        logging.info("Loading SDXL Pipelines...")

        vae = AutoencoderKL.from_pretrained(
            self.config["vae_path"],
            dtype=self.dtype,
            use_safetensors=True,
        )

        # Load Scheduler based on Config
        sched_name = self.config["generation"].get("scheduler", "EulerAncestral")
        logging.info(f"Loading Scheduler: {sched_name}")

        if sched_name == "DPMSolverMultistep":
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.config["scheduler_path"],
                subfolder="scheduler",
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        else:
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                self.config["scheduler_path"],
                subfolder="scheduler",
                timestep_spacing="trailing",
            )

        # Base Pipeline
        self.pipeline = StableDiffusionXLPAGPipeline.from_single_file(
            self.config["model_path"],
            scheduler=scheduler,
            vae=vae,
            dtype=self.dtype,
            use_safetensors=True,
            safety_checker=None,
            enable_pag=self.config["generation"].get("enable_pag", True),
        )
        if self.config["generation"].get("enable_cpu_offload", False):
            logging.info("Enabling Base Pipeline Model CPU Offload...")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)
        self.pipeline.vae.enable_tiling()

        # FreeU
        if self.config["generation"].get("enable_freeu", True):
            logging.info("Enabled FreeU.")
            self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)

        # Optimizations
        logging.info("Applying Optimizations (Channels Last, AttnProcessor)...")
        self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
        self.pipeline.unet.to(memory_format=torch.channels_last)
        self.pipeline.vae.to(memory_format=torch.channels_last)

        # Compilation
        if self.config["generation"].get("enable_compilation", False):
            try:
                logging.info("Compiling UNet (mode=reduce-overhead)...")
                self.pipeline.unet = torch.compile(
                    self.pipeline.unet, mode="reduce-overhead"
                )

                # Warmup Base
                logging.info("Warming up Base Pipeline (JIT)...")
                with torch.no_grad():
                    self.pipeline(
                        prompt="warmup",
                        num_inference_steps=1,
                        guidance_scale=self.config["generation"].get(
                            "guidance_scale", 4.0
                        ),
                        output_type="latent",
                    )
            except Exception as e:
                logging.warning(f"Compilation skipped/failed: {e}")
        else:
            logging.info("Compilation disabled in config")

        # Refiner Pipeline (Inpaint mode with SAM/Detailer, or Img2Img mode without SAM)
        if self.config.get("refiner", {}).get("enabled"):
            ref_cfg = self.config["refiner"]
            ref_model_path = ref_cfg.get("model_path", "same_as_base")
            is_shared_base = (
                ref_model_path in ("same_as_base", "", None)
                or ref_model_path == self.config.get("model_path")
            )

            if ref_cfg.get("use_sam", False):
                logging.info("Loading SDXL Refiner Pipeline (Inpaint mode)...")
                if is_shared_base:
                    logging.info(
                        "Reusing Base Pipeline components for Inpaint Pipeline (zero VRAM overhead)..."
                    )
                    self.img2img_pipeline = StableDiffusionXLInpaintPipeline.from_pipe(
                        self.pipeline
                    )
                elif str(ref_model_path).endswith(".safetensors") or (
                    isinstance(ref_model_path, str) and os.path.isfile(ref_model_path)
                ):
                    self.img2img_pipeline = (
                        StableDiffusionXLInpaintPipeline.from_single_file(
                            ref_model_path,
                            dtype=self.dtype,
                            use_safetensors=True,
                        )
                    )
                else:
                    self.img2img_pipeline = (
                        StableDiffusionXLInpaintPipeline.from_pretrained(
                            ref_model_path,
                            dtype=self.dtype,
                            use_safetensors=True,
                            variant="fp16" if self.dtype == torch.float16 else None,
                        )
                    )
            else:
                logging.info("Loading SDXL Refiner Pipeline (Img2Img mode)...")
                if is_shared_base:
                    logging.info(
                        "Reusing Base Pipeline components for Img2Img Pipeline (zero VRAM overhead)..."
                    )
                    self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(
                        self.pipeline
                    )
                elif str(ref_model_path).endswith(".safetensors") or (
                    isinstance(ref_model_path, str) and os.path.isfile(ref_model_path)
                ):
                    self.img2img_pipeline = AutoPipelineForImage2Image.from_single_file(
                        ref_model_path,
                        dtype=self.dtype,
                        use_safetensors=True,
                    )
                else:
                    self.img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
                        ref_model_path,
                        dtype=self.dtype,
                        use_safetensors=True,
                        variant="fp16" if self.dtype == torch.float16 else None,
                    )

            if not is_shared_base:
                if self.config["generation"].get("enable_cpu_offload", False):
                    logging.info("Enabling Refiner Pipeline Model CPU Offload...")
                    self.img2img_pipeline.enable_model_cpu_offload()
                else:
                    self.img2img_pipeline.to(self.device)

                # Apply optimizations
                self.img2img_pipeline.unet.set_attn_processor(AttnProcessor2_0())
                self.img2img_pipeline.unet.to(memory_format=torch.channels_last)
                self.img2img_pipeline.vae.to(memory_format=torch.channels_last)

            # Load SAM if enabled
            if self.config["refiner"].get("use_sam", False):
                logging.info("Loading SAM2 Model...")
                sam_model_name = self.config["refiner"].get(
                    "sam_model", "sam2.1_hiera_large.pt"
                )
                sam_dir = self.config["refiner"].get(
                    "sam_dir", "/usr/local/share/models/comfyui/sams"
                )
                if os.path.isabs(sam_model_name) or os.path.exists(sam_model_name):
                    sam_ckpt = sam_model_name
                else:
                    sam_ckpt = os.path.join(sam_dir, sam_model_name)

                # Custom SAM loading to use repo config directory
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                from hydra.core.global_hydra import GlobalHydra
                from hydra import initialize_config_dir, compose
                from omegaconf import OmegaConf
                from hydra.utils import instantiate
                from sam2.build_sam import _load_checkpoint

                GlobalHydra.instance().clear()
                with initialize_config_dir(config_dir=repo_root, version_base="1.2"):
                    cfg = compose(config_name="sam2.1_hiera_l")

                OmegaConf.resolve(cfg)
                sam_model = instantiate(cfg.model, _recursive_=True)
                _load_checkpoint(sam_model, sam_ckpt)
                sam_model = sam_model.to(self.device)
                sam_model.eval()

                self.sam_predictor = SAM2ImagePredictor(sam_model)

            # Compilation
            if self.config["generation"].get("enable_compilation", False):
                try:
                    logging.info("Compiling Refiner UNet (mode=reduce-overhead)...")
                    self.img2img_pipeline.unet = torch.compile(
                        self.img2img_pipeline.unet, mode="reduce-overhead"
                    )
                except Exception as e:
                    logging.warning(f"Refiner compilation skipped/failed: {e}")
            else:
                logging.info("Refiner compilation disabled in config")
        else:
            self.img2img_pipeline = None

        logging.info("Pipelines ready.")

    def _load_yolo_models(self):
        """Load YOLO models for face and body detection."""
        ref_cfg = self.config.get("refiner", {})
        if ref_cfg.get("enabled") and ref_cfg.get("use_sam", False):
            try:
                face_model_path = ref_cfg.get(
                    "yolo_face_model",
                    "/usr/local/share/models/comfyui/ultralytics/bbox/yolov12l-face.pt",
                )
                if os.path.exists(face_model_path):
                    self.face_detector = YOLO(face_model_path)
                    logging.info(f"Loaded YOLO Face model: {face_model_path}")
                else:
                    logging.warning(
                        f"YOLO Face model path not found: {face_model_path}"
                    )

                default_human_model = (
                    "/usr/local/share/models/comfyui/ultralytics/segm/person_yolov8m-seg.pt"
                    if os.path.exists(
                        "/usr/local/share/models/comfyui/ultralytics/segm/person_yolov8m-seg.pt"
                    )
                    else "/usr/local/share/models/comfyui/ultralytics/yolo12x.pt"
                )
                human_model_path = ref_cfg.get(
                    "yolo_human_model",
                    default_human_model,
                )
                if os.path.exists(human_model_path):
                    self.human_detector = YOLO(human_model_path)
                    logging.info(f"Loaded YOLO Human model: {human_model_path}")
                else:
                    logging.warning(
                        f"YOLO Human model path not found: {human_model_path}"
                    )

            except Exception as e:
                logging.error(f"Failed to load YOLO models: {e}")
                self.face_detector = None
                self.human_detector = None

    def generate(self, prompt, negative_prompt, count, seed=None):
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded.")

        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

        current_seed = self.generator.initial_seed()
        logging.info(f"Seed: {current_seed}")

        # Config extraction
        gen_cfg = self.config["generation"]

        # Embeddings
        embeds, nembeds, pooled, npooled = get_weighted_text_embeddings_sdxl(
            self.pipeline, prompt=prompt, neg_prompt=negative_prompt
        )

        # Generate
        with (
            torch.no_grad(),
            torch.autocast(self.device.type, enabled=True, dtype=self.dtype),
        ):
            result = self.pipeline(
                prompt_embeds=embeds,
                negative_prompt_embeds=nembeds,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=npooled,
                width=gen_cfg["width"],
                height=gen_cfg["height"],
                num_inference_steps=gen_cfg["steps"],
                guidance_scale=gen_cfg["guidance_scale"],
                pag_scale=gen_cfg.get("pag_scale", 3.0),
                num_images_per_prompt=count,
                generator=self.generator,
            )

        del embeds, nembeds, pooled, npooled
        self.free_memory()

        return result.images, current_seed

    def _get_target_prompt(self, ref_cfg, target, original_prompt):
        """Resolves target-specific prompt, falling back to base prompt with target anchors."""
        custom_prompt = ref_cfg.get(f"{target}_prompt")
        if custom_prompt:
            return custom_prompt
        base = ref_cfg.get("prompt") or original_prompt
        if target == "face":
            return f"{base}, detailed face, detailed eyes, sharp focus"
        elif target == "body":
            return f"{base}, highly detailed, perfect anatomy"
        elif target == "world":
            return f"{base}, detailed background, atmospheric lighting"
        return base

    def _get_target_negative_prompt(self, ref_cfg, target, original_neg_prompt):
        """Resolves target-specific negative prompt, falling back to base negative with target anchors."""
        custom_neg = ref_cfg.get(f"{target}_negative_prompt")
        if custom_neg:
            return custom_neg
        base = ref_cfg.get("negative_prompt") or original_neg_prompt
        if target == "face":
            return f"{base}, deformed eyes, crooked face, bad anatomy, ugly, distortion"
        elif target == "body":
            return f"{base}, extra limbs, malformed hands, deformed body, bad anatomy"
        elif target == "world":
            return f"{base}, blurry, artifacts, distortion"
        return base

    def _detail_world(
        self, image, person_masks, original_prompt, original_neg_prompt, ref_cfg
    ):
        """Refines the background/world by inpainting outside the person mask."""
        logging.info("Detailing World (Background)...")
        w, h = image.size
        combined_person = np.any(person_masks, axis=0)

        # Dilate person mask slightly to provide a protective boundary buffer
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        person_dilated = cv2.dilate(combined_person.astype(np.uint8), kernel) > 0
        world_bool = ~person_dilated

        world_mask = Image.fromarray((world_bool.astype(np.uint8) * 255), mode="L")
        world_mask_feathered = world_mask.filter(ImageFilter.GaussianBlur(radius=8))

        prompt = self._get_target_prompt(ref_cfg, "world", original_prompt)
        neg_prompt = self._get_target_negative_prompt(
            ref_cfg, "world", original_neg_prompt
        )
        strength = float(ref_cfg.get("world_strength", ref_cfg.get("strength", 0.25)))
        steps = int(ref_cfg.get("world_steps", ref_cfg.get("steps", 20)))
        guidance = float(
            ref_cfg.get("world_guidance_scale", ref_cfg.get("guidance_scale", 4.0))
        )

        return self.img2img_pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            mask_image=world_mask_feathered,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=self.generator,
        ).images[0]

    def _detail_body(
        self,
        image,
        person_masks,
        human_boxes,
        original_prompt,
        original_neg_prompt,
        ref_cfg,
    ):
        """Refines character body using person mask."""
        logging.info("Detailing Body...")
        w, h = image.size

        if person_masks:
            combined_person = np.any(person_masks, axis=0)
            body_mask = Image.fromarray(
                (combined_person.astype(np.uint8) * 255), mode="L"
            )
        else:
            mask_np = np.zeros((h, w), dtype=np.uint8)
            for box in human_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                mask_np[y1:y2, x1:x2] = 255
            body_mask = Image.fromarray(mask_np, mode="L")

        body_mask_feathered = body_mask.filter(ImageFilter.GaussianBlur(radius=6))

        prompt = self._get_target_prompt(ref_cfg, "body", original_prompt)
        neg_prompt = self._get_target_negative_prompt(
            ref_cfg, "body", original_neg_prompt
        )
        strength = float(ref_cfg.get("body_strength", ref_cfg.get("strength", 0.25)))
        steps = int(ref_cfg.get("body_steps", ref_cfg.get("steps", 20)))
        guidance = float(
            ref_cfg.get("body_guidance_scale", ref_cfg.get("guidance_scale", 4.0))
        )

        return self.img2img_pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            mask_image=body_mask_feathered,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=self.generator,
        ).images[0]

    def _detail_face(
        self, image, face_boxes, original_prompt, original_neg_prompt, ref_cfg
    ):
        """Refines faces using high-resolution crop inpainting."""
        logging.info(
            f"Detailing {len(face_boxes)} Face(s) via High-Resolution Crop Inpainting..."
        )
        w, h = image.size
        prompt = self._get_target_prompt(ref_cfg, "face", original_prompt)
        neg_prompt = self._get_target_negative_prompt(
            ref_cfg, "face", original_neg_prompt
        )
        strength = float(ref_cfg.get("face_strength", ref_cfg.get("strength", 0.35)))
        steps = int(ref_cfg.get("face_steps", ref_cfg.get("steps", 25)))
        guidance = float(
            ref_cfg.get("face_guidance_scale", ref_cfg.get("guidance_scale", 4.5))
        )
        crop_res = int(ref_cfg.get("face_crop_size", 768))

        refined_image = image.copy()

        for idx, box in enumerate(face_boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # 1.8x context margin around face
            size = max(bw, bh) * 1.8

            crop_x1 = max(0, int(cx - size / 2))
            crop_y1 = max(0, int(cy - size / 2))
            crop_x2 = min(w, int(cx + size / 2))
            crop_y2 = min(h, int(cy + size / 2))
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            if crop_w < 16 or crop_h < 16:
                continue

            face_crop = refined_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            rel_x1 = max(0, int(x1 - crop_x1))
            rel_y1 = max(0, int(y1 - crop_y1))
            rel_x2 = min(crop_w, int(x2 - crop_x1))
            rel_y2 = min(crop_h, int(y2 - crop_y1))

            mask_crop = Image.new("L", (crop_w, crop_h), 0)
            draw = ImageDraw.Draw(mask_crop)
            draw.ellipse([rel_x1, rel_y1, rel_x2, rel_y2], fill=255)
            mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=8))

            crop_resized = face_crop.resize(
                (crop_res, crop_res), Image.Resampling.LANCZOS
            )
            mask_resized = mask_crop.resize(
                (crop_res, crop_res), Image.Resampling.BILINEAR
            )

            refined_crop = self.img2img_pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=crop_resized,
                mask_image=mask_resized,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=self.generator,
            ).images[0]

            refined_crop_orig = refined_crop.resize(
                (crop_w, crop_h), Image.Resampling.LANCZOS
            )
            refined_image.paste(refined_crop_orig, (crop_x1, crop_y1), mask_crop)
            logging.info(f"Refined Face {idx+1}/{len(face_boxes)}.")

        return refined_image

    def _refine_global(self, image, original_prompt, original_neg_prompt):
        """Runs refinement pass using multi-target crop detailer, SAM inpaint, or img2img."""
        ref_cfg = self.config.get("refiner", {})
        if not ref_cfg.get("enabled", False):
            return image

        if not self.img2img_pipeline:
            logging.warning("Refiner pipeline not loaded, skipping refinement.")
            return image

        logging.info(
            f"Global Refinement (Strength: {ref_cfg.get('strength', 0.3)}, Steps: {ref_cfg.get('steps', 25)})..."
        )

        try:
            if not ref_cfg.get("use_sam", False):
                # Fallback to whole-image img2img without mask
                final_prompt = ref_cfg.get("prompt") or original_prompt
                final_neg = ref_cfg.get("negative_prompt") or original_neg_prompt
                return self.img2img_pipeline(
                    prompt=final_prompt,
                    negative_prompt=final_neg,
                    image=image,
                    strength=ref_cfg.get("strength", 0.35),
                    guidance_scale=ref_cfg.get("guidance_scale", 2.0),
                    num_inference_steps=ref_cfg.get("steps", 30),
                    generator=self.generator,
                ).images[0]

            # Multi-target detailing mode
            configured_targets = list(
                ref_cfg.get("targets", ["world", "body", "face"])
            )
            image_np = np.array(image)

            face_boxes = []
            if "face" in configured_targets:
                face_boxes = self._detect_faces(image_np)

            human_boxes = []
            if "body" in configured_targets or "world" in configured_targets:
                human_boxes = self._detect_humans(image_np)

            person_masks = []
            if getattr(self, "_cached_person_masks", None):
                person_masks = list(self._cached_person_masks)
            elif self.sam_predictor and human_boxes:
                self.sam_predictor.set_image(image_np)
                for box in human_boxes:
                    mask, score = self._refine_box_with_sam(box, "body")
                    if mask is not None:
                        person_masks.append(mask > 0)

            has_face = bool(face_boxes) and "face" in configured_targets
            has_body = (
                bool(person_masks) or bool(human_boxes)
            ) and "body" in configured_targets
            has_world = bool(person_masks) and "world" in configured_targets

            if not (has_face or has_body or has_world):
                logging.info("No targets detected for refinement, skipping.")
                return image

            refined_image = image.copy()

            # Execute in optimal order: world -> body -> face
            execution_order = []
            for t in ["world", "body", "face"]:
                if t in configured_targets:
                    execution_order.append(t)

            for target in execution_order:
                if target == "world" and person_masks:
                    refined_image = self._detail_world(
                        refined_image,
                        person_masks,
                        original_prompt,
                        original_neg_prompt,
                        ref_cfg,
                    )
                elif target == "body" and (person_masks or human_boxes):
                    refined_image = self._detail_body(
                        refined_image,
                        person_masks,
                        human_boxes,
                        original_prompt,
                        original_neg_prompt,
                        ref_cfg,
                    )
                elif target == "face" and face_boxes:
                    refined_image = self._detail_face(
                        refined_image,
                        face_boxes,
                        original_prompt,
                        original_neg_prompt,
                        ref_cfg,
                    )

            return refined_image

        except Exception as e:
            logging.error(f"Global Refinement failed: {e}", exc_info=True)
            return image

    def _generate_sam_mask(self, image):
        """Generate masks using YOLO detection + SAM2 segmentation for configured target areas."""
        logging.info("Generating YOLO+SAM masks...")
        try:
            # Convert PIL to numpy
            image_np = np.array(image)
            h, w = image_np.shape[:2]

            # Set image in SAM predictor
            self.sam_predictor.set_image(image_np)

            # Detect face and body regions using YOLO
            face_boxes = self._detect_faces(image_np)
            human_boxes = self._detect_humans(image_np)

            logging.info(f"Detected {len(face_boxes)} faces, {len(human_boxes)} humans")

            ref_cfg = self.config.get("refiner", {})
            configured_targets = set(ref_cfg.get("targets", ["face", "body"]))

            all_masks = []
            all_scores = []
            mask_types = []

            # Process face detections
            for box in face_boxes:
                mask, score = self._refine_box_with_sam(box, "face")
                if mask is not None:
                    all_masks.append(mask > 0)
                    all_scores.append(score)
                    mask_types.append("face")

            # Process human detections
            for box in human_boxes:
                mask, score = self._refine_box_with_sam(box, "body")
                if mask is not None:
                    all_masks.append(mask > 0)
                    all_scores.append(score)
                    mask_types.append("body")

            # Generate world/background mask if explicitly requested in targets
            if "world" in configured_targets and all_masks:
                world_mask = self._generate_world_mask(image_np, all_masks)
                all_masks.append(world_mask)
                all_scores.append(1.0)
                mask_types.append("world")

            if not all_masks:
                logging.info("No detections found, skipping SAM refinement.")
                return None

            logging.info(
                f"Generated {len(all_masks)} masks: {mask_types} with scores: {[round(s, 3) for s in all_scores]}"
            )

            # Save debug info if enabled
            if ref_cfg.get("debug_yolo", False):
                self._save_debug_masks_yolo_sam(
                    image_np, all_masks, all_scores, mask_types, face_boxes, human_boxes
                )

            # Combine only the configured targets
            combined_mask = np.zeros((h, w), dtype=bool)
            for mask, m_type in zip(all_masks, mask_types):
                if m_type in configured_targets:
                    combined_mask |= (mask > 0)

            if not np.any(combined_mask):
                logging.info(
                    f"No masks matched configured targets {configured_targets}, skipping SAM refinement."
                )
                return None

            # Convert boolean mask cleanly to 0/255 uint8 without overflow
            mask_bytes = combined_mask.astype(np.uint8) * 255
            return Image.fromarray(mask_bytes, mode="L")

        except Exception as e:
            logging.error(f"YOLO+SAM mask generation failed: {e}")
            return None

    def _detect_faces(self, image_np):
        """Detect faces using YOLO face model."""
        if not self.face_detector:
            return []

        try:
            results = self.face_detector(image_np, verbose=False)
            boxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if float(box.conf) > 0.4:
                            coords = box.xyxy[0].cpu()
                            x1, y1, x2, y2 = (
                                int(coords[0]),
                                int(coords[1]),
                                int(coords[2]),
                                int(coords[3]),
                            )

                            # Minimal expansion for face boxes
                            expand_x = int((x2 - x1) * 0.02)
                            expand_y = int((y2 - y1) * 0.02)

                            expanded_box = [
                                max(0, int(x1 - expand_x)),
                                max(0, int(y1 - expand_y)),
                                min(image_np.shape[1], int(x2 + expand_x)),
                                min(image_np.shape[0], int(y2 + expand_y)),
                            ]

                            boxes.append(expanded_box)
                            if self.config.get("refiner", {}).get("debug_yolo", False):
                                logging.info(
                                    f"Face detected: box={expanded_box}, confidence={float(box.conf):.3f}"
                                )

            return boxes
        except Exception as e:
            logging.warning(f"Face detection failed: {e}")
            return []

    def _detect_humans(self, image_np):
        """Detect bodies using YOLO person model (box and optional seg masks)."""
        if not self.human_detector:
            return []

        try:
            results = self.human_detector(image_np, verbose=False)
            boxes = []
            self._cached_person_masks = []
            h, w = image_np.shape[:2]

            for result in results:
                has_masks = result.masks is not None
                if result.boxes is not None:
                    for idx, box in enumerate(result.boxes):
                        # Filter for person class (class 0 in COCO dataset)
                        if int(box.cls) == 0 and float(box.conf) > 0.25:
                            coords = box.xyxy[0].cpu()
                            x1, y1, x2, y2 = (
                                int(coords[0]),
                                int(coords[1]),
                                int(coords[2]),
                                int(coords[3]),
                            )

                            # Expand box slightly for better body coverage
                            expand_x = int((x2 - x1) * 0.05)
                            expand_y = int((y2 - y1) * 0.05)

                            expanded_box = [
                                max(0, int(x1 - expand_x)),
                                max(0, int(y1 - expand_y)),
                                min(image_np.shape[1], int(x2 + expand_x)),
                                min(image_np.shape[0], int(y2 + expand_y)),
                            ]

                            boxes.append(expanded_box)

                            if has_masks and idx < len(result.masks.data):
                                m_np = result.masks.data[idx].cpu().numpy()
                                if m_np.shape != (h, w):
                                    m_full = (
                                        cv2.resize(
                                            m_np,
                                            (w, h),
                                            interpolation=cv2.INTER_LINEAR,
                                        )
                                        > 0.5
                                    )
                                else:
                                    m_full = m_np > 0.5
                                self._cached_person_masks.append(m_full)

                            if self.config.get("refiner", {}).get("debug_yolo", False):
                                logging.info(
                                    f"Body detected: box={expanded_box}, confidence={float(box.conf):.3f}"
                                )

            return boxes
        except Exception as e:
            logging.warning(f"Body detection failed: {e}")
            return []

    def _refine_box_with_sam(self, box, mask_type):
        """Use SAM to refine a bounding box into a precise mask."""
        try:
            # Convert box to SAM format
            sam_box = np.array([box])

            # Generate mask using SAM with different settings for faces vs bodies
            if mask_type == "face":
                # For faces: use single mask for more precision
                masks, scores, _ = self.sam_predictor.predict(
                    box=sam_box,
                    multimask_output=False,  # Single mask for faces
                )
            else:
                # For bodies: use multiple masks and pick best
                masks, scores, _ = self.sam_predictor.predict(
                    box=sam_box,
                    multimask_output=True,
                )

            # Select best mask (highest score)
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                return masks[best_idx], float(scores[best_idx])

        except Exception as e:
            logging.warning(f"SAM refinement failed for {mask_type}: {e}")

        return None, 0.0

    def _generate_world_mask(self, image_np, person_masks):
        """Generate world/background mask as inverse of person detections."""
        h, w = image_np.shape[:2]

        # Create combined person boolean mask
        combined_person = np.zeros((h, w), dtype=bool)
        for mask in person_masks:
            combined_person |= (mask > 0)

        # World mask is the inverse
        return ~combined_person

    def _save_debug_masks_yolo_sam(
        self, image_np, masks, scores, mask_types, face_boxes, human_boxes
    ):
        """Save debug information for YOLO+SAM masks."""
        try:
            import time
            import json

            # Use same directory structure as main script
            import datetime

            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            save_base = self.config.get("image_path", ".")
            debug_dir = os.path.join(save_base, date_str, "debug_masks")
            os.makedirs(debug_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save original image
            Image.fromarray(image_np).save(
                os.path.join(debug_dir, f"{timestamp}_original.png")
            )

            # Save image with detected boxes for visualization
            self._save_debug_boxes(
                image_np, face_boxes, human_boxes, debug_dir, timestamp
            )

            # Save individual masks with types and scores
            for i, (mask, score, mask_type) in enumerate(
                zip(masks, scores, mask_types)
            ):
                mask_bytes = (mask > 0).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_bytes, mode="L")
                filename = f"{timestamp}_mask_{i}_{mask_type}_score_{score:.3f}.png"
                mask_img.save(os.path.join(debug_dir, filename))
                logging.info(f"Saved debug mask: {filename}")

            # Save debug info
            debug_info = {
                "timestamp": timestamp,
                "detections": {
                    "face_boxes": face_boxes,
                    "human_boxes": human_boxes,
                    "face_count": len(face_boxes),
                    "human_count": len(human_boxes),
                },
                "masks": {
                    "count": len(masks),
                    "types": mask_types,
                    "scores": [float(s) for s in scores],
                },
            }

            with open(
                os.path.join(debug_dir, f"{timestamp}_yolo_sam_info.json"), "w"
            ) as f:
                json.dump(debug_info, f, indent=2)

            logging.info(f"Saved YOLO+SAM debug info to: {debug_dir}")

        except Exception as e:
            logging.warning(f"Failed to save YOLO+SAM debug info: {e}")

    def _save_debug_boxes(
        self, image_np, face_boxes, human_boxes, debug_dir, timestamp
    ):
        """Save image with detected boxes for visualization."""
        try:
            import cv2

            # Convert to BGR for OpenCV
            img_debug = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)

            # Draw face boxes (green)
            for box in face_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_debug,
                    "Face",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Draw body boxes (blue)
            for box in human_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    img_debug,
                    "Body",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

            # Save debug image with boxes
            cv2.imwrite(
                os.path.join(debug_dir, f"{timestamp}_detections.jpg"), img_debug
            )
            logging.info(f"Saved detection boxes: {timestamp}_detections.jpg")

        except Exception as e:
            logging.warning(f"Failed to save debug boxes: {e}")

    def process_image(self, base_image, prompt_text, negative_prompt_text):
        """Handles Refinement (Global) and Upscaling."""

        image = base_image

        # 1. Global Refinement (Img2Img)
        if self.config.get("refiner", {}).get("enabled"):
            image = self._refine_global(image, prompt_text, negative_prompt_text)

        # 2. Upscaling
        if self.config.get("upscaling", {}).get("enabled"):
            image = self.upscaler.process(image)
            self.free_memory()

        return image

    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
