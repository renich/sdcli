#! ./venv/bin/python

import gc
import os
import pathlib
import random
import re
import time
import torch
from colored import Fore, Back, Style
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from modules.utils import read_file_as_fstring
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl

# clean memory
torch.cuda.empty_cache()

# envorinment
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'


# directories
pathlib.Path('images').mkdir(parents=True, exist_ok=True)
pathlib.Path('models').mkdir(parents=True, exist_ok=True)

pipeline = StableDiffusionXLPipeline.from_single_file(
    './models/mklanANIMEHentai_nextgenV3.safetensors',
    torch_dtype = torch.float16,
    variant='fp16',
    use_safetensors = True,
    safety_checker = None,
    attn_implementation='flash_attention_2',
).to('cuda')

# optimizations
#pipeline.enable_vae_slicing()
#pipeline.enable_vae_tiling()
#pipeline.enable_sequential_cpu_offload()
pipeline.enable_model_cpu_offload()
#pipeline.enable_xformers_memory_efficient_attention()
pipeline.disable_xformers_memory_efficient_attention()

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
#pipeline.unet = torch.compile(pipeline.unet, mode='max-autotune', fullgraph=True)
#pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode='max-autotune', fullgraph=True)


# scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

# cleanup
#os.system('find images/ -type f -iname "*.png" -delete')

# pipeline
for i in range(100):
    # prompt
    ## create a dictionary to store wildcards
    prompt_data = {
        'action': '',
        'attire': '',
        'character': '',
        'footwear': '',
        'mood': '',
        'place': '',
        'pov': '',
    }

    # get the keys from the prompt_data dictionary
    keys = list(prompt_data.keys())

    # iterate through each key and fill it with a random value from the corresponding file
    for key in keys:
        filename = f'wildcards/{key}s.txt'
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        prompt_data[key] = random.choice(lines)

    # get prompt
    filename = 'prompts/default.prompt'
    result = read_file_as_fstring(filename, **prompt_data)

    if result:
        prompt = result

    # get negative prompt
    filename = 'prompts/default.nprompt'
    result = read_file_as_fstring(filename, **prompt_data)

    if result:
        nprompt = result

    # print prompts
    print(f'''

{Style.bold}{Fore.white}{Back.green}prompt{Style.reset}:
{prompt}

{Style.bold}{Fore.white}{Back.red}negative prompt{Style.reset}:
{nprompt}
    ''')

    embeds, nembeds, pooled, npooled = get_weighted_text_embeddings_sdxl(
        pipeline,
        prompt = prompt,
        neg_prompt = nprompt
    )

    # pipeline
    batch = pipeline(
        prompt_embeds = embeds,
        negative_prompt_embeds = nembeds,
        pooled_prompt_embeds = pooled,
        negative_pooled_prompt_embeds = npooled,
        width = 1024,
        height = 1024,
        guidance_scale = 3.5,
        num_inference_steps = 50,
        num_images_per_prompt = 5,
    )

    # save
    for idx, image in enumerate(batch.images):
        ## date, time and name
        date = time.strftime('%Y-%m-%d')
        daytime = time.strftime('%H%M%S')
        name = f'images/{date}/{time.strftime("%Y%m%d")}-{daytime}.png'

        ## crear dir
        pathlib.Path(f'images/{date}').mkdir(parents=True, exist_ok=True)

        ## save
        image.save(name)

        ## sleep
        time.sleep(0.5)

        ## open eof
        try:
            os.system(f'/usr/bin/eog -w {name} &')
        except:
            pass

    # garbage collection
    del embeds, nembeds, pooled, npooled
    gc.collect()
    torch.cuda.empty_cache()

