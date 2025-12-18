from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import re

def read_file_as_fstring(filename, **kwargs):
    with open(filename, 'r') as file:
        lines = file.readlines()

    non_comment_lines = [line for line in lines if not line.startswith('#')]

    # Join the remaining lines into a single string with newlines
    fstring_text = '\n'.join(non_comment_lines)

    # Replace multiple spaces after commas with a single space
    fstring_text = re.sub(r',\s+', ', ', fstring_text)

    try:
        result = eval(f'f"""{fstring_text}"""', globals(), {**kwargs, **locals()})
    except Exception as e:
        print(f"Error evaluating f-string: {e}")
        return None

    return result

def embed_metadata_to_png(image_path, metadata):
  """
  Embeds metadata into a PNG image.

  Args:
      image_path: The path to the PNG image file.
      metadata: A dictionary containing the metadata to embed.
  """

  img = Image.open(image_path)
  png_info = PngInfo()

  for key, value in metadata.items():
    png_info.add_text(key, value)

  img.save(image_path, pnginfo=png_info)

