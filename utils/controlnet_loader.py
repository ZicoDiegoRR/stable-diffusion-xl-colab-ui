from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
from transformers import pipeline as pipe
from diffusers import ControlNetModel
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

# Clearing VRAM
def controlnet_flush(
    pipeline,
    controlnets,
    loaded_controlnet_model,
    images,
    controlnets_scale,
    loaded_pipeline,
    loaded_model,
):
    cn_reset = ""
    cn_reset_sanitized_list = [element for element in loaded_controlnet_model if element]
    for weight in cn_reset_sanitized_list:
        if cn_reset_sanitized_list.index(weight) == (len(cn_reset_sanitized_list) - 1) and len(cn_reset_sanitized_list) > 1:
            cn_reset += f"and {weight} ControlNets"
        elif len(cn_reset_sanitized_list) == 1:
            cn_reset += f"{weight} ControlNet"
        else:
            cn_reset += f"{weight}, " if len(cn_reset_sanitized_list) == 3 else f"{weight} "
    print(f"You previously activated the {cn_reset}. Because of this, the pipeline must be reloaded to free up some VRAM.")
    print("Flushing...")
    
    to_be_reset = [
        controlnets, 
        loaded_controlnet_model, 
        images, 
        controlnets_scale, 
        loaded_pipeline,
        loaded_model,
    ]
    for value in to_be_reset:
        if isinstance(value, list):
            for element in value:
                del element
                element = None
        else:
            if value:
                del value
                value = None
    if pipeline:
        del pipeline.controlnet
        pipeline.controlnet = None
    torch.cuda.empty_cache()
    gc.collect()

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
    controlnets,
    loaded_controlnet_model,
    images,
    controlnets_scale,
    loaded_pipeline,
    loaded_model,
    
):
    # Flushing if ControlNet is loaded but unused
    if (not Canny and controlnets[0]) or (not Depth_Map and controlnets[1]) or (not Open_Pose and controlnets[2]): 
        controlnet_flush(
            pipeline,
            controlnets,
            loaded_controlnet_model,
            images,
            controlnets_scale,
            loaded_pipeline,
            loaded_model,
        )

    # Loading ControlNet
    if (Canny or Depth_Map or Open_Pose) and (Canny_link or Depthmap_Link or Openpose_Link):
        # Handling Canny
        if Canny and Canny_link is not None:
            if "canny" not in loaded_controlnet_model:
                print("Loading Canny...")
                loaded_controlnet_model[0] = "canny"
                controlnets[0] = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    use_safetensors=True, 
                    low_cpu_mem_usage=True,
                    variant="fp16"
                )
            print("Converting image with Canny Edge Detection...")
            c_img = Canny_link
            image_canny = np.array(c_img)
            image_canny = cv2.Canny(image_canny, minimum_canny_threshold, maximum_canny_threshold)
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
            if "depth" not in loaded_controlnet_model:
                print("Loading Depth Map...")
                loaded_controlnet_model[1] = "depth"
                controlnets[1] = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    use_safetensors=True, 
                    low_cpu_mem_usage=True,
                    variant="fp16"
                ).to("cuda")
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
            if "openpose" not in loaded_controlnet_model:
                print("Loading Open Pose...")
                loaded_controlnet_model[2] = "openpose"
                controlnets[2] = ControlNetModel.from_pretrained(
                    "xinsir/controlnet-openpose-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True
                ).to("cuda")
            print("Converting image with Open Pose...")
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cpu")
            openpose_width, openpose_height = Openpose_Link.size
            openpose_image = openpose(Openpose_Link)
            images[2] = openpose_image.resize((1024, 1024))
            controlnets_scale[2] = Open_Pose_Strength
            print("Open Pose is done.")
            display(make_image_grid([Openpose_Link, openpose_image.resize((openpose_width, openpose_height))], rows=1, cols=2))
            del openpose

        if len([cn for cn in controlnets if cn]) == 1:
            for model in loaded_controlnet_model:
                if model:
                    single_cn_index = loaded_controlnet_model.index(model)
            pipeline.controlnet = controlnets[single_cn_index]
        else:
            pipeline.controlnet = [cn for cn in controlnets if cn]
            
        torch.cuda.empty_cache()
        gc.collect()
