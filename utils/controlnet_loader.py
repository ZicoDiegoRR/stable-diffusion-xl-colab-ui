from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
from transformers import pipeline as pipe
from diffusers import ControlNetUnionModel
from IPython.display import display
from PIL import Image as ImagePIL
import numpy as np
import torch
import json
import cv2
import os

# Loading the path of the latest generated images
def load_last(filename, type):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data.get(type, None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Selecting the image based on secret keywords
def controlnet_path_selector(path, type, base_path):
    last_generation_loading = os.path.join(base_path, "last_generation.json")
    try:
        if path == "inpaint":
            cn_path = load_last(last_generation_loading, 'inpaint')                     
        elif path == "controlnet":
            cn_path = load_last(last_generation_loading, 'controlnet')
        elif not path:
            cn_path = load_last(last_generation_loading, 'text2img')
        else:
            cn_path = path
        cn_image = load_image(cn_path)
        pipeline_type = "controlnet"
    except Exception as e:
        if path == "inpaint":
            cn_path = "last-generated Inpainting image."                     
        elif path == "controlnet":
            cn_path = "last-generated ControlNet image."
        elif not path:
            cn_path = "last-generated Text-to-image image."
        else:
            cn_path = path
        print(f"Couldn't load {cn_path}. Reason: {e}")
        cn_image = ""
        pipeline_type = type
    return cn_image, pipeline_type

def flush(index, images, controlnets_scale):
    for element in [images, controlnets_scale]:
        element[index] = None

# Loading the ControlNet if activated
def load(
    pipeline,
    Canny,
    Canny_link,
    minimum_canny_threshold,
    maximum_canny_threshold,
    Canny_Strength,
    Depth_Map,
    Depthmap_Link,
    Depth_Strength,
    Open_Pose,
    Openpose_Link,
    Open_Pose_Strength,
    controlnet,
    images,
    controlnets_scale,
    get_image_class,
):
    # Deleting images and scales for inactivated ControlNet
    if not Canny and images[0]:
        flush(index, images, controlnets_scale)
    elif not Depth_Map and images[1]:
        flush(index, images, controlnets_scale)
    elif not Open_Pose and images[2]:
        flush(index, images, controlnets_scale)
    
    # Loading ControlNet
    if (Canny or Depth_Map or Open_Pose) and (Canny_link or Depthmap_Link or Openpose_Link):
        # Loading the weight if hasn't been loaded
        if not controlnet:
            controlnet = ControlNetUnionModel.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16)
            
        # Handling Canny
        if Canny and Canny_link is not None:
            print("Converting image with Canny Edge Detection...")
            canny_image = get_image_class.get_canny(Canny_link)
            canny_width, canny_height = Canny_link.size
            
            print("Canny Edge Detection is complete.")
            display(make_image_grid([Canny_link, canny_image.resize((canny_width, canny_height))], rows=1, cols=2))
            
            images[0] = canny_image.resize((1024, 1024))
            controlnets_scale[0] = Canny_Strength
            
        # Handling Depth Map
        if Depth_Map and Depthmap_Link is not None:
            print("Converting image with Depth Map...")
            image_depth = Depthmap_Link.resize((1024, 1024))
            
            depth_map = get_image_class.get_depth(image_depth).unsqueeze(0).half().to("cpu")
            depth_map_display = get_image_class.get_depth(image_depth, "display")
            
            depth_width, depth_height = Depthmap_Link.size
            controlnets_scale[1] = Depth_Strength
            images[1] = depth_map
            
            print("Depth Map is complete.")
            display(make_image_grid([Depthmap_Link, depth_map_display.resize((depth_width, depth_height))], rows=1, cols=2))
            
        # Handling Open Pose
        if Open_Pose and Openpose_Link is not None:
            print("Converting image with Open Pose...")
            openpose_width, openpose_height = Openpose_Link.size
            openpose_image = get_image_class.get_openpose(Openpose_Link)
            
            images[2] = openpose_image.resize((1024, 1024))
            controlnets_scale[2] = Open_Pose_Strength
            
            print("Open Pose is done.")
            display(make_image_grid([Openpose_Link, openpose_image.resize((openpose_width, openpose_height))], rows=1, cols=2))
