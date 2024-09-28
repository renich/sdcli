#!./venv/bin/python

import os
import pathlib
import random
import re
import time
import torch
from colored import Fore, Back, Style
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

# environment
os.environ['PYTORCH_ROCM_ARCH'] = "gfx1030"
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# directories
pathlib.Path('images').mkdir(parents=True, exist_ok=True)

# pipeline init
pipeline = StableDiffusionXLPipeline.from_single_file(
    './models/mklanANIMEHentai_nextgenV3.safetensors',
    torch_dtype = torch.float16,
    use_safetensors = True,
    safety_checker = None,
    variant="fp16",
).to("cuda")

# prompt parser init
compel = Compel(
    tokenizer = [pipeline.tokenizer, pipeline.tokenizer_2],
    text_encoder = [pipeline.text_encoder, pipeline.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled = [False, True],
)

# optimizations
pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()
#pipeline.enable_sequential_cpu_offload()
pipeline.enable_model_cpu_offload()
pipeline.disable_xformers_memory_efficient_attention()
#pipeline.enable_xformers_memory_efficient_attention()

## pytorch 2.0
#torch.backends.cuda.matmul.allow_tf32 = True
pipeline.unet.set_attn_processor(AttnProcessor2_0())
pipeline.vae.set_attn_processor(AttnProcessor2_0())
#pipeline.unet.set_default_attn_processor()
#pipeline.vae.set_default_attn_processor()

## compile
#torch._inductor.config.conv_1x1_as_mm = True
#torch._inductor.config.coordinate_descent_tuning = True
#torch._inductor.config.epilogue_fusion = False
#torch._inductor.config.coordinate_descent_check_all_directions = True
#pipeline.unet.to(memory_format=torch.channels_last)
#pipeline.vae.to(memory_format=torch.channels_last)
#pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
#pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

# scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

# cleanup
#os.system('find images/ -type f -iname "*.png" -delete')

# pipeline
for i in range(100):
    # prompt
    ## wildcards
    action = random.choice(open('wildcards/actions').read().splitlines())
    attire = random.choice(open('wildcards/attire').read().splitlines())
    character = random.choice(open('wildcards/characters').read().splitlines())
    mood = random.choice(open('wildcards/moods').read().splitlines())
    place = random.choice(open('wildcards/places').read().splitlines())
    pov = random.choice(open('wildcards/povs').read().splitlines())
    footwear = random.choice(open('wildcards/footwear').read().splitlines())

    ## prompts
    prompt = ''
    with open('prompts/default.prompt', 'r', encoding = 'utf-8') as promptfile:
        for line in promptfile:
            if line.startswith('#'):
                continue
            prompt += f'{line.strip()}'.format(**locals())

    nprompt = ''
    with open('prompts/default.nprompt', 'r', encoding = 'utf-8') as promptfile:
        for line in promptfile:
            if line.startswith('#'):
                continue
            nprompt += f'{line.strip()}'.format(**locals())

    # print prompts
    print(f'''

{Style.bold}{Fore.white}{Back.green}prompt{Style.reset}:
{prompt}

{Style.bold}{Fore.white}{Back.red}negative prompt{Style.reset}:
{nprompt}
    ''')

    ## process prompt
    conditioning, pooled = compel(prompt)
    nconditioning, npooled = compel(nprompt)

    # clean cache
    torch.cuda.empty_cache()

    # pipeline
    batch = pipeline(
        prompt_embeds = conditioning,
        negative_prompt_embeds = nconditioning,
        pooled_prompt_embeds = pooled,
        negative_pooled_prompt_embeds = npooled,
        width = 1024,
        height = 1024,
        guidance_scale = 3.5,
        num_inference_steps = 30,
        num_images_per_prompt = 1,
        attn_implementation="flash_attention_2",
    )

    # save
    for idx, image in enumerate(batch.images):
        name = "images/{}.png".format(time.strftime('%Y%m%d-%H%M%S'))
        image.save(name)
        time.sleep(0.5)
