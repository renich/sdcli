#!/root/src/venv/bin/python

from diffusers import AutoPipelineForText2Image
import torch
import os

os.system('rm -f *.png')

# Load the base model
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')

# Load the uncensored LoRA weights
pipeline.load_lora_weights('enhanceaiteam/Flux-uncensored', weight_name='lora.safetensors')

# Generate an image with an uncensored NSFW prompt
prompt = """

Proffesional photo of a young sexy woman with a super muscular physique exercising at the park. She has an incredibly big ass.

"""


for n in range(100):
    image = pipeline(
            prompt.rstrip(), 
            #num_inference_steps=50
        ).images[0]
    image.save("image-{}.png".format(n))

