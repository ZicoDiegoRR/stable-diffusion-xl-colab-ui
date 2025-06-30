from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os

def run(pipe, img, hires_values, gen_args): # [lanczos or realesrgan, factor, denoising strength]
    os.makedirs("/content/hires", exist_ok=True)
    img.save("/content/hires/temp.png")

    vals = hires_values
    width, height = img.size
    try:
        if vals[0] == "LANCZOS":
            image = img.resize((width*vals[1], height*vals[1]), Image.LANCZOS)
        else:
            vals[0].hires_execute("/content/hires/temp.png", vals[1])
            image = Image.open("/content/hires/upscale.png")

        img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)

        hires_image = img2img(**gen_args, image=image, strength=vals[2]).images[0]
        
    except Exception as e:
        print("Couldn't apply Hires.Fix to the image. Returning the original image instead...")
        return img

    return hires_image
