from StableDiffusionXLColabUI.utils import (
    embeddings_loader, 
    image_saver,
    lora_loader,
    pipeline_selector,
    run_generation,
    vae_loader,
)
from diffusers import (DDPMScheduler, 
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler, 
    KDPM2DiscreteScheduler, 
    KDPM2AncestralDiscreteScheduler, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler, 
    HeunDiscreteScheduler, 
    LMSDiscreteScheduler, 
    DEISMultistepScheduler, 
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    PNDMScheduler,
)
from PIL import Image as ImagePIL
from controlnet_aux import OpenposeDetector
from compel import Compel, ReturnedEmbeddingsType
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image, make_image_grid
from huggingface_hub import login
import ipywidgets as widgets
import numpy as np
import random
import json
import time
import os

loaded_model = ""
loaded_pipeline = ""
vae_current = ""
loaded_controlnet_model = [None] * 3
controlnets = [None] * 3
images = [None] * 3
controlnets_scale = [None] * 3

def save_param(path, data):
    with open(path, 'w') as file:
        json.dump(data, file)

def load_last(filename, type):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data.get(type, None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def restart(new, old):
    print(f"New model is found. Your previous one ({old}) is different than your new one ({new}).")
    print("Restarting the runtime is necessary to load the new one.")
    time.sleep(2)
    print("Restarting the runtime...")
    time.sleep(0.5)
    os.kill(os.getpid(), 9)

def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map

# Only for display in output, nothing crazy
def get_depth_map_display(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return image

def run(values_in_list, lora, embeddings, ip, hf_token, civit_token, ui, seed_list, dictionary, widgets_change):
    # Initialization
    pipeline_type = ""
    if len(values_in_list) == 15:
        pipelie_type = "text2img"
        selected_tab_for_pipeline = 0
    elif len(values_in_list) == 17:
        pipelie_type = "img2img"
        selected_tab_for_pipeline = 1
    elif len(values_in_list) == 26:
        pipelie_type = "controlnet"
        selected_tab_for_pipeline = 2
    elif len(values_in_list) == 19:
        pipelie_type = "inpaint"
        selected_tab_for_pipeline = 3

    if not seed_list[1] and seed_list[0].value == -1:
        generator_seed = random.randint(1, 1000000000000)
    else:
        generator_seed = seed_list[0].value

    base_path = "/content/gdrive/MyDrive" if os.path.exists("/content/gdrive/MyDrive") else "/content"

    # Gathering values
    Prompt = values_in_list[0]
    Negative_Prompt = values_in_list[1]
    Model = values_in_list[2]

    Width = values_in_list[3]
    Height = values_in_list[4]
    Steps = values_in_list[5]
    Scale = values_in_list[6]
    Clip_Skip = values_in_list[7]

    Scheduler = values_in_list[8]
    Karras = values_in_list[9]
    V_Prediction = values_in_list[10]
    SGMUniform = values_in_list[11]
    Rescale_betas_to_zero_SNR = values_in_list[12]
    
    VAE_Link = values_in_list[13]
    VAE_Config = values_in_list[14]

    Reference_Image = values_in_list[15] if pipeline_type == "img2img" else None
    Denoising_Strength = values_in_list[16] if pipeline_type == "img2img" else None

    LoRA_URLs = lora[0]
    Weight_Scale = lora[1]

    Textual_Inversion_URLs = embeddings[0]
    Textual_Inversion_Tokens = embeddings[1]

    Canny_Link = values_in_list[15] if pipeline_type == "controlnet" else None
    minimum_canny_threshold = values_in_list[16] if pipeline_type == "controlnet" else None
    maximum_canny_threshold = values_in_list[17] if pipeline_type == "controlnet" else None
    Canny = values_in_list[18] if pipeline_type == "controlnet" else None
    Canny_Strength = values_in_list[19] if pipeline_type == "controlnet" else None

    DepthMap_Link = values_in_list[20] if pipeline_type == "controlnet" else None
    Depth_Map = values_in_list[21] if pipeline_type == "controlnet" else None
    Depth_Strength = values_in_list[22] if pipeline_type == "controlnet" else None

    OpenPose_Link = values_in_list[23] if pipeline_type == "controlnet" else None
    Open_Pose = values_in_list[24] if pipeline_type == "controlnet" else None
    Open_Pose_Strength = values_in_list[25] if pipeline_type == "controlnet" else None

    Inpainting_Image = values_in_list[15] if pipeline_type == "inpaint" else None
    Mask_Image = values_in_list[16] if pipeline_type == "inpaint" else None
    Inpainting = values_in_list[17] if pipeline_type == "inpaint" else None
    Inpainting_Strength = values_in_list[18] if pipeline_type == "inpaint" else None

    IP_Image_Link = ip[0]
    IP_Adapter_Strength = ip[1]
    IP_Adapter = ip[2]

    HF_Token = hf_token
    Civit_Token = civit_token

    # Reusing the old logic
    if hf_token:
      login(hf_token)

    last_generation_loading = os.path.join(base_path, "last_generation.json")
    if Canny and selected_tab_for_pipeline == 2:
        if Canny_Link == "inpaint":
            Canny_link = load_last(last_generation_loading, 'inpaint')
        elif Canny_Link == "controlnet":
            Canny_link = load_last(last_generation_loading, 'controlnet')
        elif not Canny_Link:
            Canny_link = load_last(last_generation_loading, 'text2img')
        else:
            Canny_link = Canny_Link
        if Canny_link or os.path.exists(Canny_link):
            pipeline_type = "controlnet"
    else:
        Canny_link = ""

    if Depth_Map and selected_tab_for_pipeline == 2:
        if DepthMap_Link == "inpaint":
            Depthmap_Link = load_last(last_generation_loading, 'inpaint')
        elif DepthMap_Link == "controlnet":
            Depthmap_Link = load_last(last_generation_loading, 'controlnet')
        elif not DepthMap_Link:
            Depthmap_Link = load_last(last_generation_loading, 'text2img')
        else:
            Depthmap_Link = DepthMap_Link
        if Depthmap_Link or os.path.exists(Depthmap_Link):
            pipeline_type = "controlnet"
    else:
        Depthmap_Link = ""

    if Open_Pose and selected_tab_for_pipeline == 2:
        if OpenPose_Link == "inpaint":
            Openpose_Link = load_last(last_generation_loading, 'inpaint')
        elif OpenPose_Link == "controlnet":
            Openpose_Link = load_last(last_generation_loading, 'controlnet')
        elif not OpenPose_Link:
            Openpose_Link = load_last(last_generation_loading, 'text2img')
        else:
            Openpose_Link = OpenPose_Link
        if Openpose_Link or os.path.exists(Openpose_Link):
            pipeline_type = "controlnet"
    else:
        Openpose_Link = ""

    active_inpaint = False
    if Inpainting and selected_tab_for_pipeline == 3:
        if Canny or Depth_Map or Open_Pose:
            raise TypeError("You checked both ControlNet and Inpainting, which will cause incompatibility issues during your run. As of now, there's no alternative way to merge StableDiffusionXLControlNetPipeline and StableDiffusionXLInpaintingPipeline without causing any issues. Perhaps you want to use only one of them?")
        if not Mask_Image:
            raise ValueError("You checked Inpainting while you're leaving Mask_Image empty. Mask_Image is required for Inpainting!")
        if Inpainting_Image == "pre-generated text2image image":
            inpaint_img = load_last(last_generation_loading, 'text2img')
        elif Inpainting_Image == "pre-generated controlnet image":
            inpaint_img = load_last(last_generation_loading, 'controlnet')
        elif Inpainting_Image == "previous inpainting image":
            inpaint_img = load_last(last_generation_loading, 'inpaint')
        else:
            inpaint_image = Inpainting_Image
        if inpaint_img is not None and os.path.exists(inpaint_img):
            pipeline_type = "inpaint"
            inpaint_image = load_image(inpaint_img).resize((1024, 1024))
            mask_image = load_image(Mask_Image).resize((1024, 1024))
            active_inpaint = True
            display(make_image_grid([inpaint_image, mask_image], rows=1, cols=2))

    if Reference_Image and selected_tab_for_pipeline == 1:
      ref_image = load_image(Reference_Image)
      if ref_image or os.path.exists(ref_image):
        pipeline_type = "img2img"
    else:
      ref_image = ""

    if not IP_Image_Link and IP_Adapter != "None":
        print(f"You selected {IP_Adapter}, but left the IP_Image_Link empty. Skipping IP-Adapter...")
        IP_Adapter = "None"
    if selected_tab_for_pipeline == 0 or (not Canny_link and not Depthmap_Link and not Openpose_Link and not active_inpaint and not ref_image):
        pipeline_type = "text2img"
        if selected_tab_for_pipeline != 0:
            print("No reference image. Defaulting to Text-to-Image...")

    save_param(f"{base_path}/Saved Parameters/main_parameters.json", dictionary)
    if os.path.exists(os.path.join(f"{base_path}", "parameters.json")):
      os.remove(os.path.join(f"{base_path}", "parameters.json"))

    # Check if the current pipeline and model are the same as the previous ones
    global loaded_model, loaded_pipeline
    if loaded_model and loaded_model != Model:
        restart(Model, loaded_model)
    if loaded_pipeline and loaded_pipeline != pipeline_type:
        restart(pipeline_type, loaded_pipeline)

    # Logic to handle ControlNet and/or MultiControlNets
    global controlnets, loaded_controlnet_model, images, controlnets_scale
    if Canny and Canny_link is not None:
        if "canny" not in loaded_controlnet_model:
          global canny_model
          canny_model = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True)
          loaded_controlnet_model[0] = "canny"
          controlnets[0] = canny_model
        print("🏞️ | Converting image with Canny Edge Detection...")
        c_img = load_image(Canny_link)
        image_canny = np.array(c_img)
        image_canny = cv2.Canny(image_canny, minimum_canny_threshold, maximum_canny_threshold)
        image_canny = image_canny[:, :, None]
        image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
        canny_image = ImagePIL.fromarray(image_canny)
        print("✅ | Canny Edge Detection is complete.")
        time.sleep(1)
        display(make_image_grid([c_img, canny_image.resize((1024, 1024))], rows=1, cols=2))
        images[0] = canny_image.resize((1024, 1024))
        controlnets_scale[0] = Canny_Strength

    if Depth_Map and Depthmap_Link is not None:
        if "depth" not in loaded_controlnet_model:
          global depthmap_model
          loaded_controlnet_model[1] = "depth"
          depthmap_model = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True).to("cuda")
          controlnets[1] = depthmap_model
        print("🏞️ | Converting image with Depth Map...")
        image_depth = load_image(Depthmap_Link).resize((1024, 1024))
        depth_estimator = pipe("depth-estimation")
        depth_map = get_depth_map(image_depth, depth_estimator).unsqueeze(0).half().to("cpu")
        images[1] = depth_map
        depth_map_display = ImagePIL.fromarray(get_depth_map_display(image_depth, depth_estimator))
        print("✅ | Depth Map is complete.")
        controlnets_scale[1] = Depth_Strength
        time.sleep(1)
        display(make_image_grid([image_depth, depth_map_display], rows=1, cols=2))

    if Open_Pose and Openpose_Link is not None:
        if "openpose" not in loaded_controlnet_model:
          global openpose, openpose_model
          loaded_controlnet_model[2] = "openpose"
          openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cpu")
          openpose_model = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
          controlnets[2] = openpose_model
        print("🏞️ | Converting image with Open Pose...")
        image_openpose = load_image(Openpose_Link)
        openpose_image = openpose(image_openpose)
        images[2] = openpose_image.resize((1024, 1024))
        print("✅ | Open Pose is done.")
        controlnets_scale[2] = Open_Pose_Strength
        display(make_image_grid([image_openpose, openpose_image.resize((1024, 1024))], rows=1, cols=2))

    global vae_current
    if VAE_Link and VAE_Link != vae_current:
        vae = vae_loader.load_vae(vae_current, VAE_Link, VAE_Config, widgets_change[0], HF_Token, Civit_Token)
    elif not VAE_Link:
        vae = None

    global pipeline
    pipeline = pipeline_selector.load_pipeline(
        Model, 
        widgets_change[1], 
        controlnets=controlnets, 
        active_inpaint=active_inpaint, 
        vae=vae, 
        hf_token=HF_Token, 
        civit_token=Civit_Token
    )
    
    if IP_Adapter != "None":
        pipeline.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to("cuda")

    if not loaded_model:
        loaded_model = Model
    if not loaded_pipeline:
        loaded_pipeline = pipeline_type
