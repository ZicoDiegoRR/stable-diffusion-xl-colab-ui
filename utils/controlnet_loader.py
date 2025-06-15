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
import gc
import os

# Loading the path of the latest generated images
def load_last(filename, type):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data.get(type, None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

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

# Converting image into a depth map used for ControlNet
def get_depth_map(image, depth_estimator, output):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    if output == "display":
        return ImagePIL.fromarray(image) # For display

    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map

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
    
):
    # Loading ControlNet
    if (Canny or Depth_Map or Open_Pose) and (Canny_link or Depthmap_Link or Openpose_Link):
        if not controlnet:
            controlnet = ControlNetUnionModel.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16)
            
        # Handling Canny
        if Canny and Canny_link is not None:
            print("Converting image with Canny Edge Detection...")
            image_canny = cv2.Canny(np.array(Canny_link), minimum_canny_threshold, maximum_canny_threshold)
            image_canny = image_canny[:, :, None]
            image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
            canny_image = ImagePIL.fromarray(image_canny)
            canny_width, canny_height = c_img.size
            print("Canny Edge Detection is complete.")
            display(make_image_grid([c_img, canny_image.resize((1024, 1024))], rows=1, cols=2))
            images[0] = canny_image.resize((canny_width, canny_height))
            controlnets_scale[0] = Canny_Strength
            
        # Handling Depth Map
        if Depth_Map and Depthmap_Link is not None:
            print("Converting image with Depth Map...")
            depth_estimator = pipe("depth-estimation", device="cpu")
            image_depth = Depthmap_Link.resize((1024, 1024))
            depth_map = get_depth_map(image_depth, depth_estimator, "depth").unsqueeze(0).half().to("cpu")
            depth_map_display = get_depth_map(image_depth, depth_estimator, "display")
            images[1] = depth_map
            depth_width, depth_height = Depthmap_Link.size
            controlnets_scale[1] = Depth_Strength
            print("Depth Map is complete.")
            display(make_image_grid([Depthmap_Link, depth_map_display.resize((depth_width, depth_height))], rows=1, cols=2))
            del depth_estimator
            
        # Handling Open Pose
        if Open_Pose and Openpose_Link is not None:
            print("Converting image with Open Pose...")
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cpu")
            openpose_width, openpose_height = Openpose_Link.size
            openpose_image = openpose(Openpose_Link)
            images[2] = openpose_image.resize((1024, 1024))
            controlnets_scale[2] = Open_Pose_Strength
            print("Open Pose is done.")
            display(make_image_grid([Openpose_Link, openpose_image.resize((openpose_width, openpose_height))], rows=1, cols=2))
            del openpose
            
        torch.cuda.empty_cache()
        gc.collect()
