from diffusers import StableDiffusionXLImg2ImgPipeline
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
import time
import os

def run(pipe, img, hires_values, gen_args): # [lanczos or realesrgan, factor, denoising strength]
    # Create the folder and save the image temporarily
    os.makedirs("/content/hires", exist_ok=True)
    img.save("/content/hires/temp.png")

    # Obtaining the original image size
    vals = hires_values
    width, height = img.size

    # Adding the output widget
    output = widgets.Output()
    display(output)

    # Keeping the output inside of the widget
    with output:
        try:
            # Upscale and resize
            print("Upscaling...")
            if vals[0] == "LANCZOS":
                image = img.resize((width*vals[1], height*vals[1]), Image.LANCZOS)
            else:
                vals[0].hires_execute("/content/hires/temp.png", vals[1])
                image = Image.open("/content/hires/upscale.png")
    
            # Initiate the Image-to-image pipeline
            print("Initiating Image-to-image pipeline...")
            img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    
            # Feed the upscaled image to the Image-to-image pipeline
            print("Refining image with Image-to-image pipeline...")
            hires_image = img2img(**gen_args, image=image, strength=vals[2]).images[0]
            
        except Exception as e:
            # Return the original image instead if the upscaling or the Image-to-image fails
            print(f"Couldn't apply Hires.Fix to the image. Reason: {e}\nReturning the original image instead...")
            hires_image = img

    # Clearing the generation output
    time.sleep(4)
    output.clear_output()
    del output

    return hires_image
