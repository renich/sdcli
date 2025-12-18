import gc
import logging
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLPAGPipeline,
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoPipelineForImage2Image
)
from diffusers.models.attention_processor import AttnProcessor2_0
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
from modules.components import Upscaler

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

        # Components
        self.upscaler = None

    def _get_device(self):
        if torch.cuda.is_available():
            if torch.version.hip:
                logging.info("Device: AMD GPU (ROCm)")
            else:
                logging.info("Device: NVIDIA GPU (CUDA)")
            return torch.device('cuda')
        return torch.device('cpu')

    def load_resources(self):
        """Loads all necessary models and pipelines."""
        self._load_pipelines()
        self.upscaler = Upscaler(self.config, self.device)

    def _load_pipelines(self):
        logging.info("Loading SDXL Pipelines...")
        
        vae = AutoencoderKL.from_pretrained(
            self.config['vae_path'],
            dtype=self.dtype,
            use_safetensors=True,
        )
        
        # Load Scheduler based on Config
        sched_name = self.config['generation'].get('scheduler', 'EulerAncestral')
        logging.info(f"Loading Scheduler: {sched_name}")
        
        if sched_name == "DPMSolverMultistep":
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.config['scheduler_path'],
                subfolder='scheduler',
                dtype=self.dtype,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        else:
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                self.config['scheduler_path'],
                subfolder='scheduler',
                dtype=self.dtype,
                timestep_spacing="trailing"
            )

        # Base Pipeline
        self.pipeline = StableDiffusionXLPAGPipeline.from_single_file(
            self.config['model_path'],
            scheduler=scheduler,
            vae=vae,
            dtype=self.dtype,
            use_safetensors=True,
            safety_checker=None,
            enable_pag=self.config['generation'].get('enable_pag', True),
        )
        self.pipeline.to(self.device)
        self.pipeline.vae.enable_tiling()

        # FreeU
        if self.config['generation'].get('enable_freeu', True):
            logging.info("Enabled FreeU.")
            self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)

        # Optimizations
        logging.info("Applying Optimizations (Channels Last, AttnProcessor)...")
        self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
        self.pipeline.unet.to(memory_format=torch.channels_last)
        self.pipeline.vae.to(memory_format=torch.channels_last)

        # Compilation
        try:
            logging.info("Compiling UNet (mode=reduce-overhead)...")
            self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead")
            
            # Warmup Base
            logging.info("Warming up Base Pipeline (JIT)...")
            with torch.no_grad():
                self.pipeline(prompt="warmup", num_inference_steps=1, guidance_scale=0.0, output_type="latent")
        except Exception as e:
            logging.warning(f"Compilation skipped/failed: {e}")

        # Refiner Pipeline (Inpaint mode with SAM)
        if self.config.get('refiner', {}).get('enabled'):
            logging.info("Loading SDXL Refiner Pipeline (Inpaint mode)...")
            self.img2img_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.config['refiner']['model_path'],
                dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            self.img2img_pipeline.to(self.device)

            # Apply same optimizations
            self.img2img_pipeline.unet.set_attn_processor(AttnProcessor2_0())
            self.img2img_pipeline.unet.to(memory_format=torch.channels_last)
            self.img2img_pipeline.vae.to(memory_format=torch.channels_last)

            # Load SAM if enabled
            if self.config['refiner'].get('use_sam', False):
                logging.info("Loading SAM2 Model...")
                sam_ckpt = f"/usr/local/share/models/comfyui/sams/{self.config['refiner']['sam_model']}"
                sam_config = f"/usr/local/share/models/comfyui/sams/sam2.1_hiera_l.yaml"  # Assuming config file exists
                sam_model = build_sam2(sam_config, sam_ckpt, device=self.device)
                self.sam_predictor = SAM2ImagePredictor(sam_model)

            # Compilation
            try:
                logging.info("Compiling Refiner UNet (mode=reduce-overhead)...")
                self.img2img_pipeline.unet = torch.compile(self.img2img_pipeline.unet, mode="reduce-overhead")
            except Exception as e:
                logging.warning(f"Refiner compilation skipped/failed: {e}")
        else:
            self.img2img_pipeline = None
        
        logging.info("Pipelines ready.")

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
        gen_cfg = self.config['generation']
        
        # Embeddings
        embeds, nembeds, pooled, npooled = get_weighted_text_embeddings_sdxl(
            self.pipeline, prompt=prompt, neg_prompt=negative_prompt
        )

        # Generate
        with torch.no_grad(), torch.autocast(self.device.type, enabled=True, dtype=self.dtype):
            result = self.pipeline(
                prompt_embeds=embeds,
                negative_prompt_embeds=nembeds,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=npooled,
                width=gen_cfg['width'],
                height=gen_cfg['height'],
                num_inference_steps=gen_cfg['steps'],
                guidance_scale=gen_cfg['guidance_scale'],
                pag_scale=gen_cfg.get('pag_scale', 3.0),
                num_images_per_prompt=count,
                generator=self.generator
            )

        del embeds, nembeds, pooled, npooled
        self.free_memory()

        return result.images, current_seed

    def _refine_global(self, image, original_prompt, original_neg_prompt):
        """Runs a SAM-guided inpaint refinement pass."""
        ref_cfg = self.config.get('refiner', {})
        if not ref_cfg.get('enabled', False):
            return image

        logging.info(f"Global Refinement (Strength: {ref_cfg['strength']}, Steps: {ref_cfg['steps']})...")

        # Use refiner-specific prompts, not the original prompts
        final_prompt = ref_cfg.get('prompt', original_prompt)
        final_neg = ref_cfg.get('negative_prompt', original_neg_prompt)

        # Generate mask using SAM if enabled
        mask = None
        if ref_cfg.get('use_sam', False) and self.sam_predictor:
            mask = self._generate_sam_mask(image)

        # If no mask or SAM failed, create full mask for full inpaint
        if mask is None:
            mask = Image.new("L", (image.width, image.height), 255)  # White mask = inpaint all

        # Get embeddings
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
        ref_embeds, ref_nembeds, ref_pooled, ref_npooled = get_weighted_text_embeddings_sdxl(
            self.img2img_pipeline, prompt=final_prompt, neg_prompt=final_neg
        )

        try:
            refined = self.img2img_pipeline(
                prompt_embeds=ref_embeds,
                negative_prompt_embeds=ref_nembeds,
                pooled_prompt_embeds=ref_pooled,
                negative_pooled_prompt_embeds=ref_npooled,
                image=image,
                mask_image=mask,
                strength=ref_cfg.get('strength', 0.35),
                guidance_scale=ref_cfg.get('guidance_scale', 7.5),
                num_inference_steps=ref_cfg.get('steps', 30),
                generator=self.generator
            ).images[0]
            del ref_embeds, ref_nembeds, ref_pooled, ref_npooled
            self.free_memory()
            return refined
        except Exception as e:
            logging.error(f"Global Refinement failed: {e}")
            return image

    def _generate_sam_mask(self, image):
        """Generate combined mask from SAM detections for body, face, and world areas."""
        try:
            # Convert PIL to numpy
            image_np = np.array(image)

            # Set image in predictor
            self.sam_predictor.set_image(image_np)

            # Generate automatic masks
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True,
            )

            # Filter masks by confidence
            threshold = self.config['refiner'].get('sam_confidence_threshold', 0.7)
            valid_masks = masks[scores > threshold]

            if len(valid_masks) == 0:
                logging.warning("No valid SAM masks found above threshold")
                return None

            # Combine all valid masks (union)
            combined_mask = np.zeros_like(valid_masks[0], dtype=np.uint8)
            for mask in valid_masks:
                combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)

            # Convert to PIL
            mask_pil = Image.fromarray(combined_mask, mode='L')
            return mask_pil

        except Exception as e:
            logging.error(f"SAM mask generation failed: {e}")
            return None

    def process_image(self, base_image, prompt_text, negative_prompt_text):
        """Handles Refinement (Global) and Upscaling."""
        
        image = base_image

        # 1. Global Refinement (Img2Img)
        if self.config.get('refiner', {}).get('enabled'):
            image = self._refine_global(image, prompt_text, negative_prompt_text)

        # 2. Upscaling
        if self.config.get('upscaling', {}).get('enabled'):
            image = self.upscaler.process(image)
            self.free_memory()
            
        return image

    def free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
