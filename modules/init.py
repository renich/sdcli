import gc
import logging
import torch
from diffusers import (
    StableDiffusionXLPAGPipeline, 
    AutoencoderKL, 
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler,
    AutoPipelineForImage2Image
)
from diffusers.models.attention_processor import AttnProcessor2_0
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
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        
        # Load Scheduler based on Config
        sched_name = self.config['generation'].get('scheduler', 'EulerAncestral')
        logging.info(f"Loading Scheduler: {sched_name}")
        
        if sched_name == "DPMSolverMultistep":
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.config['scheduler_path'],
                subfolder='scheduler',
                torch_dtype=self.dtype,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        else:
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                self.config['scheduler_path'],
                subfolder='scheduler',
                torch_dtype=self.dtype,
                timestep_spacing="trailing"
            )

        # Base Pipeline
        self.pipeline = StableDiffusionXLPAGPipeline.from_single_file(
            self.config['model_path'],
            scheduler=scheduler,
            vae=vae,
            torch_dtype=self.dtype,
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

        # Img2Img Pipeline
        logging.info("Initializing Img2Img Pipeline (Global Refiner)...")
        self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(
            self.pipeline,
            torch_dtype=self.dtype,
            use_safetensors=True
        )
        self.img2img_pipeline.to(self.device)
        
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
        """Runs a global Img2Img refinement pass."""
        ref_cfg = self.config.get('refiner', {})
        if not ref_cfg.get('enabled', False):
            return image

        logging.info(f"Global Refinement (Strength: {ref_cfg['strength']}, Steps: {ref_cfg['steps']})...")
        
        refine_prompt_add = ref_cfg.get('prompt', "")
        refine_neg_add = ref_cfg.get('negative_prompt', "")
        
        final_prompt = original_prompt
        if refine_prompt_add:
            final_prompt = f"{original_prompt}, {refine_prompt_add}"
            
        final_neg = original_neg_prompt
        if refine_neg_add:
            final_neg = f"{original_neg_prompt}, {refine_neg_add}"

        try:
            refined = self.img2img_pipeline(
                prompt=final_prompt,
                negative_prompt=final_neg,
                image=image,
                strength=ref_cfg.get('strength', 0.35),
                guidance_scale=ref_cfg.get('guidance_scale', 7.5),
                num_inference_steps=ref_cfg.get('steps', 30),
                generator=self.generator
            ).images[0]
            self.free_memory()
            return refined
        except Exception as e:
            logging.error(f"Global Refinement failed: {e}")
            return image

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
