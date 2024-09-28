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
