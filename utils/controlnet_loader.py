from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
from transformers import pipeline as pipe
from diffusers import ControlNetModel
from PIL import Image as ImagePIL
import numpy as np
import torch
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
def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map

# Only for display in output for depth map, nothing crazy
def get_depth_map_display(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return image

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
        pipeline.controlnet = None
    torch.cuda.empty_cache()
    gc.collect()

def load(
    pipeline,
    Canny,
    Depth_Map,
    Open_Pose,
    Canny_link,
    Depthmap_Link,
    Openpose_Link,
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
            controlnets,
            pipeline,
            controlnets,
            loaded_controlnet_model,
            images,
            controlnets_scale,
            loaded_pipeline,
            loaded_model,
        )

    # Loading ControlNet
    if pipeline_type == "controlnet" and (Canny or Depth_Map or Open_Pose) and (Canny_link or Depthmap_Link or Openpose_Link):
        # Handling Canny
        if Canny and Canny_link is not None:
            if "canny" not in loaded_controlnet_model:
                print("Loading Canny...")
                loaded_controlnet_model[0] = "canny"
                controlnets[0] = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    use_safetensors=True, 
                    low_cpu_mem_usage=True
                )
            print("Converting image with Canny Edge Detection...")
            c_img = Canny_link
            image_canny = np.array(c_img)
            image_canny = cv2.Canny(image_canny, minimum_canny_threshold, maximum_canny_threshold)
            image_canny = image_canny[:, :, None]
            image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
            canny_image = ImagePIL.fromarray(image_canny)
            print("Canny Edge Detection is complete.")
            display(make_image_grid([c_img, canny_image.resize((1024, 1024))], rows=1, cols=2))
            images[0] = canny_image.resize((1024, 1024))
            controlnets_scale[0] = Canny_Strength
            
        # Handling Depth Map
        if Depth_Map and Depthmap_Link is not None:
            if "depth" not in loaded_controlnet_model:
                print("Loading Depth Map...")
                loaded_controlnet_model[1] = "depth"
                controlnets[1] = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    use_safetensors=True, 
                    low_cpu_mem_usage=True
                ).to("cuda")
            print("Converting image with Depth Map...")
            depth_estimator = pipe("depth-estimation", device="cpu")
            image_depth = Depthmap_Link.resize((1024, 1024))
            depth_map = get_depth_map(image_depth, depth_estimator).unsqueeze(0).half().to("cpu")
            images[1] = depth_map
            controlnets_scale[1] = Depth_Strength
            depth_map_display = ImagePIL.fromarray(get_depth_map_display(image_depth, depth_estimator))
            print("Depth Map is complete.")
            display(make_image_grid([image_depth, depth_map_display], rows=1, cols=2))
            del depth_estimator
            
        # Handling Open Pose
        if Open_Pose and Openpose_Link is not None:
            if "openpose" not in loaded_controlnet_model:
                print("Loading Open Pose...")
                loaded_controlnet_model[2] = "openpose"
                controlnets[2] = ControlNetModel.from_pretrained(
                    "thibaud/controlnet-openpose-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True
                ).to("cuda")
            print("Converting image with Open Pose...")
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cpu")
            openpose_image = openpose(Openpose_Link)
            images[2] = openpose_image.resize((1024, 1024))
            controlnets_scale[2] = Open_Pose_Strength
            print("Open Pose is done.")
            display(make_image_grid([Openpose_Link, openpose_image.resize((1024, 1024))], rows=1, cols=2))
            del openpose

        if len(controlnets) == 1:
            for model in loaded_controlnet_model:
                if model:
                    single_cn_index = loaded_controlnet_model.index(model)
            pipeline.controlnet = controlnets[single_cn_index]
        else:
            pipeline.controlnet = [cn for cn in controlnets if cn]
            
        torch.cuda.empty_cache()
        gc.collect()
