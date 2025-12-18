import gc
import logging
import os
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
            return torch.device('cuda')
        return torch.device('cpu')

    def load_resources(self):
        """Loads all necessary models and pipelines."""
        self._load_pipelines()
        self._load_yolo_models()
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
        if self.config['generation'].get('enable_compilation', False):
            try:
                logging.info("Compiling UNet (mode=reduce-overhead)...")
                self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead")

                # Warmup Base
                logging.info("Warming up Base Pipeline (JIT)...")
                with torch.no_grad():
                    self.pipeline(prompt="warmup", num_inference_steps=1, guidance_scale=0.0, output_type="latent")
            except Exception as e:
                logging.warning(f"Compilation skipped/failed: {e}")
        else:
            logging.info("Compilation disabled in config")

        # Refiner Pipeline (Inpaint mode with SAM, or Img2Img mode without SAM)
        if self.config.get('refiner', {}).get('enabled'):
            if self.config['refiner'].get('use_sam', False):
                logging.info("Loading SDXL Refiner Pipeline (Inpaint mode)...")
                self.img2img_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                    self.config['refiner']['model_path'],
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    variant="fp16" if self.dtype == torch.float16 else None,
                )
            else:
                logging.info("Loading SDXL Refiner Pipeline (Img2Img mode)...")
                self.img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
                    self.config['refiner']['model_path'],
                    torch_dtype=self.dtype,
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

                # Custom SAM loading to use local config
                from hydra.core.global_hydra import GlobalHydra
                from hydra import initialize_config_dir, compose
                from omegaconf import OmegaConf
                from hydra.utils import instantiate
                from sam2.build_sam import _load_checkpoint

                GlobalHydra.instance().clear()
                with initialize_config_dir(config_dir='/home/renich/Projects/hf/sdcli'):
                    cfg = compose(config_name='sam2.1_hiera_l')

                OmegaConf.resolve(cfg)
                sam_model = instantiate(cfg.model, _recursive_=True)
                _load_checkpoint(sam_model, sam_ckpt)
                sam_model = sam_model.to(self.device)
                sam_model.eval()

                self.sam_predictor = SAM2ImagePredictor(sam_model)

            # Compilation
            if self.config['generation'].get('enable_compilation', False):
                try:
                    logging.info("Compiling Refiner UNet (mode=reduce-overhead)...")
                    self.img2img_pipeline.unet = torch.compile(self.img2img_pipeline.unet, mode="reduce-overhead")
                except Exception as e:
                    logging.warning(f"Refiner compilation skipped/failed: {e}")
            else:
                logging.info("Refiner compilation disabled in config")
        else:
            self.img2img_pipeline = None
        
        logging.info("Pipelines ready.")

    def _load_yolo_models(self):
        """Load YOLO models for face and body detection."""
        if self.config.get('refiner', {}).get('enabled') and self.config['refiner'].get('use_sam', False):
            try:
                # Load YOLO12 face detection model
                face_model_path = "/usr/local/share/models/comfyui/ultralytics/bbox/yolov12l-face.pt"
                self.face_detector = YOLO(face_model_path)
                logging.info(f"Loaded YOLO12 Face model: {face_model_path}")
                
                # Load YOLO12 general model for human detection
                human_model_path = "/usr/local/share/models/comfyui/ultralytics/yolo12x.pt"
                self.human_detector = YOLO(human_model_path)
                logging.info(f"Loaded YOLO12 Human model: {human_model_path}")
                
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
        """Runs refinement pass using inpaint or img2img."""
        ref_cfg = self.config.get('refiner', {})
        if not ref_cfg.get('enabled', False):
            return image

        logging.info(f"Global Refinement (Strength: {ref_cfg['strength']}, Steps: {ref_cfg['steps']})...")

        # Use refiner-specific prompts, not the original prompts
        final_prompt = ref_cfg.get('prompt', original_prompt)
        final_neg = ref_cfg.get('negative_prompt', original_neg_prompt)

        try:
            if ref_cfg.get('use_sam', False) and self.sam_predictor:
                # Use inpaint with SAM-generated mask
                mask = self._generate_sam_mask(image)
                if mask is None:
                    mask = Image.new("L", (image.width, image.height), 255)

                refined = self.img2img_pipeline(
                    prompt=final_prompt,
                    negative_prompt=final_neg,
                    image=image,
                    mask_image=mask,
                    strength=ref_cfg.get('strength', 0.35),
                    guidance_scale=ref_cfg.get('guidance_scale', 2.0),
                    num_inference_steps=ref_cfg.get('steps', 30),
                    generator=self.generator
                ).images[0]
            else:
                # Use img2img without mask
                refined = self.img2img_pipeline(
                    prompt=final_prompt,
                    negative_prompt=final_neg,
                    image=image,
                    strength=ref_cfg.get('strength', 0.35),
                    guidance_scale=ref_cfg.get('guidance_scale', 2.0),
                    num_inference_steps=ref_cfg.get('steps', 30),
                    generator=self.generator
                ).images[0]

            return refined
        except Exception as e:
            logging.error(f"Global Refinement failed: {e}")
            return image

    def _generate_sam_mask(self, image):
        """Generate masks using YOLO detection + SAM2 segmentation for face, body, and world areas."""
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

            # Generate SAM masks for each detection
            all_masks = []
            all_scores = []
            mask_types = []

            # Process face detections
            for box in face_boxes:
                mask, score = self._refine_box_with_sam(box, "face")
                if mask is not None:
                    all_masks.append(mask)
                    all_scores.append(score)
                    mask_types.append("face")

            # Process human detections  
            for box in human_boxes:
                mask, score = self._refine_box_with_sam(box, "body")
                if mask is not None:
                    all_masks.append(mask)
                    all_scores.append(score)
                    mask_types.append("body")

            # Generate world/background mask (inverse of all person detections)
            if all_masks:
                world_mask = self._generate_world_mask(image_np, all_masks)
                all_masks.append(world_mask)
                all_scores.append(1.0)  # World mask gets highest score
                mask_types.append("world")

            if not all_masks:
                logging.warning("No detections found, falling back to full inpaint")
                return Image.new("L", (image.width, image.height), 255)

            # Convert to numpy array
            masks_np = np.array(all_masks)
            scores_np = np.array(all_scores)

            logging.info(f"Generated {len(all_masks)} masks: {mask_types} with scores: {scores_np}")

            # Save debug info if enabled
            if self.config.get('refiner', {}).get('debug_yolo', False):
                self._save_debug_masks_yolo_sam(image_np, masks_np, scores_np, mask_types, face_boxes, human_boxes)

            # Combine all masks (union of all detections)
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in all_masks:
                combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)

            return Image.fromarray(combined_mask, mode='L')

        except Exception as e:
            logging.error(f"YOLO+SAM mask generation failed: {e}")
            return None

    def _detect_faces(self, image_np):
        """Detect faces using YOLO12 face model."""
        if not self.face_detector:
            return []
        
        try:
            results = self.face_detector(image_np, verbose=False)
            boxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if box.conf > 0.4:  # Slightly lower confidence threshold
                            coords = box.xyxy[0].cpu()
                            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                            
                            # Slightly expand face box for better coverage
                            expand = int((x2 - x1) * 0.05)  # 5% expansion
                            expanded_box = [
                                max(0, int(x1 - expand)), 
                                max(0, int(y1 - expand)),
                                min(image_np.shape[1], int(x2 + expand)), 
                                min(image_np.shape[0], int(y2 + expand))
                            ]
                            
                            boxes.append(expanded_box)
                            if self.config.get('refiner', {}).get('debug_yolo', False):
                                logging.info(f"Face detected: box={expanded_box}, confidence={float(box.conf):.3f}")
            
            return boxes
        except Exception as e:
            logging.warning(f"Face detection failed: {e}")
            return []

    def _detect_humans(self, image_np):
        """Detect bodies using YOLO12 general model (person class)."""
        if not self.human_detector:
            return []
        
        try:
            results = self.human_detector(image_np, verbose=False)
            boxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Filter for person class (class 0 in COCO dataset) with lower threshold for stylized characters
                        if int(box.cls) == 0 and box.conf > 0.25:  # Even lower threshold for better body detection
                            coords = box.xyxy[0].cpu()
                            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                            
                            # Expand box slightly for better body coverage
                            expand_x = int((x2 - x1) * 0.1)  # 10% horizontal expansion
                            expand_y = int((y2 - y1) * 0.1)  # 10% vertical expansion
                            
                            expanded_box = [
                                max(0, int(x1 - expand_x)), 
                                max(0, int(y1 - expand_y)),
                                min(image_np.shape[1], int(x2 + expand_x)), 
                                min(image_np.shape[0], int(y2 + expand_y))
                            ]
                            
                            boxes.append(expanded_box)
                            if self.config.get('refiner', {}).get('debug_yolo', False):
                                logging.info(f"Body detected: box={expanded_box}, confidence={box.conf:.3f}")
            
            return boxes
        except Exception as e:
            logging.warning(f"Body detection failed: {e}")
            return []

    def _refine_box_with_sam(self, box, mask_type):
        """Use SAM to refine a bounding box into a precise mask."""
        try:
            # Convert box to SAM format
            sam_box = np.array([box])
            
            # Generate mask using SAM
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
        
        # Create combined person mask
        combined_person = np.zeros((h, w), dtype=np.uint8)
        for mask in person_masks:
            combined_person = np.maximum(combined_person, mask.astype(np.uint8))
        
        # World mask is the inverse
        world_mask = 255 - combined_person
        
        return world_mask

    def _save_debug_masks_yolo_sam(self, image_np, masks, scores, mask_types, face_boxes, human_boxes):
        """Save debug information for YOLO+SAM masks."""
        try:
            import time
            import json
            
            # Use same directory structure as main script
            import datetime
            date_str = datetime.datetime.now().strftime('%Y-%m-%d')
            save_base = self.config.get('image_path', '.')
            debug_dir = os.path.join(save_base, date_str, 'debug_masks')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Save original image
            Image.fromarray(image_np).save(os.path.join(debug_dir, f'{timestamp}_original.png'))
            
            # Save image with detected boxes for visualization
            self._save_debug_boxes(image_np, face_boxes, human_boxes, debug_dir, timestamp)
            
            # Save individual masks with types and scores
            for i, (mask, score, mask_type) in enumerate(zip(masks, scores, mask_types)):
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
                filename = f'{timestamp}_mask_{i}_{mask_type}_score_{score:.3f}.png'
                mask_img.save(os.path.join(debug_dir, filename))
                logging.info(f"Saved debug mask: {filename}")
            
            # Save debug info
            debug_info = {
                'timestamp': timestamp,
                'detections': {
                    'face_boxes': face_boxes,
                    'human_boxes': human_boxes,
                    'face_count': len(face_boxes),
                    'human_count': len(human_boxes)
                },
                'masks': {
                    'count': len(masks),
                    'types': mask_types,
                    'scores': [float(s) for s in scores]
                }
            }
            
            with open(os.path.join(debug_dir, f'{timestamp}_yolo_sam_info.json'), 'w') as f:
                json.dump(debug_info, f, indent=2)
                
            logging.info(f"Saved YOLO+SAM debug info to: {debug_dir}")
            
        except Exception as e:
            logging.warning(f"Failed to save YOLO+SAM debug info: {e}")

    def _save_debug_boxes(self, image_np, face_boxes, human_boxes, debug_dir, timestamp):
        """Save image with detected boxes for visualization."""
        try:
            import cv2
            
            # Convert to BGR for OpenCV
            img_debug = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)
            
            # Draw face boxes (green)
            for box in face_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_debug, 'Face', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw body boxes (blue)
            for box in human_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(img_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_debug, 'Body', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Save debug image with boxes
            cv2.imwrite(os.path.join(debug_dir, f'{timestamp}_detections.jpg'), img_debug)
            logging.info(f"Saved detection boxes: {timestamp}_detections.jpg")
            
        except Exception as e:
            logging.warning(f"Failed to save debug boxes: {e}")

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
