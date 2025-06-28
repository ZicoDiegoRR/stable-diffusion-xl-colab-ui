from IPython.display import display
import time
import os
import re

def name_generate_and_save(image, img, i, image_save_path, generated_image_raw_filename):
    if len(generated_image_raw_filename) > 255:
        truncate_length = 251 - len(f"_{i}") if len(image.images) != 1 else 251
        generated_image_filename = generated_image_raw_filename[:truncate_length]  
    else:
        generated_image_filename = generated_image_raw_filename

    if len(image.images) != 1:
        generated_image_savefile = f"{image_save_path}/{generated_image_filename}_{i}.png"
    else:
        generated_image_savefile = f"{image_save_path}/{generated_image_filename}.png"
        
    img.save(generated_image_savefile)
    return generated_image_savefile
        

def save_image(image, prompt_for_name, prefix, scheduler, seed, base_path):
    current_time = time.localtime()
    formatted_time = time.strftime("[%H-%M-%S %B %d, %Y]", current_time)
    if prefix == "[Text-to-Image]":
        image_save_path = f"{base_path}/Text2Img"
    elif prefix == "[ControlNet]":
        image_save_path = f"{base_path}/ControlNet"
    elif prefix == "[Inpainting]":
        image_save_path = f"{base_path}/Inpainting"
    else:
        image_save_path = f"{base_path}/Img2Img"
    os.makedirs(image_save_path, exist_ok=True)
  
    sub_prompt = re.sub(r"[<>:\"/\\|?*]", "_", prompt_for_name)
    split_prompt = re.split(r"\s*,\s*", sub_prompt)
    prompt_name = " ".join(split_prompt)
    generated_image_raw_filename = f"{prefix} {formatted_time} {prompt_name}"
    for i, img in enumerate(image.images):
        generated_image_savefile = name_generate_and_save(image, img, i, image_save_path, generated_image_raw_filename)
        
        display(img)
        print(f"Scheduler: {''.join(scheduler)}")
        print(f"Seed: {seed}")
        print(f"Image is saved at {generated_image_savefile}.\n")
