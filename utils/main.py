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
    ControlNetModel,
)
from PIL import Image as ImagePIL
from controlnet_aux import OpenposeDetector
from compel import Compel, ReturnedEmbeddingsType
from transformers import CLIPVisionModelWithProjection, pipeline as pipe
from diffusers.utils import load_image, make_image_grid
from IPython.display import display, clear_output
from huggingface_hub import login
import ipywidgets as widgets
import numpy as np
import random
import torch
import json
import time
import os

# Variables to avoid loading the same model or pipeline twice
pipeline = None
loaded_model = ""
loaded_pipeline = ""
vae_current = ""
loaded_controlnet_model = [None] * 3
controlnets = [None] * 3
images = [None] * 3
controlnets_scale = [None] * 3

# Saving the set parameters
def save_param(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

# Saving the path of the latest generated images
def save_last(filename, data, type):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        if type == "[Text-to-Image]":
            existing_data['text2img'] = data
        elif type == "[ControlNet]":
            existing_data['controlnet'] = data
        elif type == "[Inpainting]":
            existing_data['inpaint'] = data
        with open(filename, 'w') as file:
            json.dump(existing_data, file, indent=4)
    except Exception as e:
        print(f"Error occurred: {e}")

# Loading the path of the latest generated images
def load_last(filename, type):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data.get(type, None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Restarting the runtime if the selected model or pipeline is different compared to the loaded ones
def restart(new, old, type):
    if type == "model":
        warning_restart = f"You inputted a new model ({new}), which is different than the old one ({old})."
    elif type == "pipeline":
        warning_restart = f"You changed the pipeline from {old} to {new}."
    print(f"{warning_restart} Restarting is required to free some memory and avoid OutOfMemory error. Restarting...")
    time.sleep(0.5)
    os.kill(os.getpid(), 9)

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

# Selecting image path for ControlNet
def controlnet_path_selector(path, type):
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
        print(f"Couldn't load {path}.")
        cn_image = ""
        pipeline_type = type
    return cn_image, pipeline_type

# Initializing image generation
def run(values_in_list, lora, embeddings, ip, hf_token, civit_token, ui, seed_list, dictionary, widgets_change):
    # Initialization
    pipeline_type = ""
    if len(values_in_list) == 15:
        pipeline_type = "text2img"
        selected_tab_for_pipeline = 0
    elif len(values_in_list) == 17:
        pipeline_type = "img2img"
        selected_tab_for_pipeline = 1
    elif len(values_in_list) == 26:
        pipeline_type = "controlnet"
        selected_tab_for_pipeline = 2
    elif len(values_in_list) == 19:
        pipeline_type = "inpaint"
        selected_tab_for_pipeline = 3

    if not seed_list[1] and seed_list[0].value == -1:
        generator_seed = random.randint(1, 1000000000000)
    elif seed_list[1] and seed_list[0].value > -1:
        generator_seed = seed_list[0].value
    elif seed_list[0].value < -1:
        print("Seed cannot be less than -1. Randomizing the seed instead...")
        generator_seed = random.randint(1, 1000000000000)
    else:
        generator_seed = random.randint(1, 1000000000000)

    seed_list[0].value = generator_seed

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

    # Logging in to HF hub if Hugging Face's token is not empty
    if hf_token:
      login(hf_token)

    # Selecting image and pipeline
    last_generation_loading = os.path.join(base_path, "last_generation.json")
    if Canny and selected_tab_for_pipeline == 2:
        Canny_link, pipeline_type = controlnet_path_selector(Canny_Link, pipeline_type)
    else:
        Canny_link = ""

    if Depth_Map and selected_tab_for_pipeline == 2:
        Depthmap_Link, pipeline_type = controlnet_path_selector(DepthMap_Link, pipeline_type)
    else:
        Depthmap_Link = ""

    if Open_Pose and selected_tab_for_pipeline == 2:
        Openpose_Link, pipeline_type = controlnet_path_selector(OpenPose_Link, pipeline_type)
    else:
        Openpose_Link = ""

    active_inpaint = False
    if Inpainting and selected_tab_for_pipeline == 3:        
        if not Mask_Image:
            print("You checked Inpainting while you're leaving mask image empty. Mask image is required for Inpainting.")
        else:
            if Inpainting_Image == "pre-generated text2image image":
                inpaint_img = load_last(last_generation_loading, 'text2img')
            elif Inpainting_Image == "pre-generated controlnet image":
                inpaint_img = load_last(last_generation_loading, 'controlnet')
            elif Inpainting_Image == "previous inpainting image":
                inpaint_img = load_last(last_generation_loading, 'inpaint')
            else:
                inpaint_img = Inpainting_Image
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
        ref_image = None

    if not IP_Image_Link and IP_Adapter != "None":
        print(f"You selected {IP_Adapter}, but left the IP_Image_Link empty. Skipping IP-Adapter...")
        IP_Adapter = "None"
    if selected_tab_for_pipeline == 0 or (not Canny_link and not Depthmap_Link and not Openpose_Link and not active_inpaint and not ref_image):
        pipeline_type = "text2img"
        if selected_tab_for_pipeline != 0:
            print("No reference image was inputted. Defaulting to Text-to-Image...")

    # Saving the set parameters (first phase)
    save_param(f"{base_path}/Saved Parameters/main_parameters.json", dictionary)

    # Deleting old save if exists
    if os.path.exists(os.path.join(f"{base_path}", "parameters.json")):
        os.remove(os.path.join(f"{base_path}", "parameters.json"))

    # Logic to handle ControlNet and/or MultiControlNets
    global controlnets, loaded_controlnet_model, images, controlnets_scale
    if Canny and Canny_link is not None:
        if "canny" not in loaded_controlnet_model:
          global canny_model
          canny_model = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True)
          loaded_controlnet_model[0] = "canny"
          controlnets[0] = canny_model
        print("🏞️ | Converting image with Canny Edge Detection...")
        c_img = Canny_link
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
        image_depth = Depthmap_Link.resize((1024, 1024))
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
        image_openpose = Openpose_Link
        openpose_image = openpose(image_openpose)
        images[2] = openpose_image.resize((1024, 1024))
        print("✅ | Open Pose is done.")
        controlnets_scale[2] = Open_Pose_Strength
        display(make_image_grid([image_openpose, openpose_image.resize((1024, 1024))], rows=1, cols=2))

    # Handling pipeline and model loading
    global pipeline, loaded_model, loaded_pipeline
    pipeline, model_name = pipeline_selector.load_pipeline(
        pipeline,
        Model, 
        widgets_change[1], 
        loaded_model, 
        loaded_pipeline,
        pipeline_type,
        controlnets=controlnets, 
        active_inpaint=active_inpaint, 
        hf_token=HF_Token, 
        civit_token=Civit_Token,
        base_path=base_path
    )

    # Handling VAE
    global vae_current
    if VAE_Link and (VAE_Link != vae_current or not vae_current):
        vae, loaded_vae = vae_loader.load_vae(
            vae_current, 
            VAE_Link, 
            VAE_Config, 
            widgets_change[0], 
            HF_Token, 
            Civit_Token,
            base_path=base_path
        )
        vae_current = loaded_vae
        if vae is not None:
            pipeline.vae = vae

    # Using a custom image encoder if IP-Adapter is True
    if IP_Adapter != "None":
        pipeline.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to("cuda")

    # Assigning new values 
    loaded_model = model_name
    loaded_pipeline = pipeline_type

    # Xformer, generator, and safety checker
    pipeline.enable_xformers_memory_efficient_attention()
    generator = torch.Generator("cpu").manual_seed(generator_seed)
    pipeline.safety_checker = None

    # Handling schedulers
    Prediction_type = "v_prediction" if V_Prediction else "epsilon"
    scheduler_args = {"prediction_type": Prediction_type,
                           "use_karras_sigmas": Karras,
                           "rescale_betas_zero_snr": Rescale_betas_to_zero_SNR
                           }
    if SGMUniform:
      scheduler_args["timestep_spacing"] = "trailing"
    Scheduler_used = ["", f"{Scheduler} ", "", "", ""]
    Scheduler_used[0] = "V-Prediction " if Prediction_type == "v_prediction" else ""
    Scheduler_used[2] = "Karras " if Karras else ""
    Scheduler_used[3] = "SGMUniform " if SGMUniform else ""
    Scheduler_used[4] = "with zero SNR betas rescaling" if Rescale_betas_to_zero_SNR else ""
    if Scheduler == "DPM++ 2M":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM++ 2M SDE":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", **scheduler_args)
    elif Scheduler == "DPM++ SDE":
        pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM2":
        pipeline.scheduler = KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM2 a":
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DDPM":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Euler a":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Heun":
        pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "LMS":
        pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DEIS":
        pipeline.scheduler = DEISMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "UniPC":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "PNDM":
        pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    # Using prompt weighting with Compel
    compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2], text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True], truncate_long_prompts=False)
    conditioning, pooled = compel([Prompt, Negative_Prompt])

    # Loading LoRA if not empty
    if LoRA_URLs:
        lora_loader.process(
            pipeline, 
            lora[0], 
            lora[1], 
            widgets_change[2], 
            HF_Token, 
            Civit_Token,
            base_path=base_path
        )
        torch.cuda.empty_cache()

    # Loading embeddings if not empty
    if Textual_Inversion_URLs:
        embeddings_loader.process(
            pipeline, 
            embeddings[0], 
            embeddings[1], 
            widgets_change[3], 
            HF_Token, 
            Civit_Token,
            base_path=base_path
        )
        torch.cuda.empty_cache()

    # Handling IP-Adapter
    image_embeds = None
    if IP_Adapter != "None" and IP_Image_Link:
        # Loading the images
        adapter_image = []
        simple_Url = [word for word in re.split(r"\s*,\s*", IP_Image_Link) if word]
        for link in simple_Url:
            adapter_image.append(load_image(link))

        # Creating the display
        adapter_display = [element for element in adapter_image]
        if len(adapter_image) % 3 == 0:
            row = int(len(adapter_image)/3)
        else:
            row = int(len(adapter_image)/3) + 1
            for i in range(3*row - len(adapter_image)):
                adapter_display.append(load_image("https://huggingface.co/IDK-ab0ut/BFIDIW9W29NFJSKAOAOXDOKERJ29W/resolve/main/placeholder.png"))
        print("Image(s) for IP-Adapter:")
        display(make_image_grid([element.resize((1024, 1024)) for element in adapter_display], rows=row, cols=3))

        # Loading the images to the IP-Adapter
        image_embeds = [adapter_image]
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=IP_Adapter, low_cpu_mem_usage=True)
        pipeline.set_ip_adapter_scale(IP_Adapter_Strength)
    torch.cuda.empty_cache()

    # Generating image
    prefix, image = run_generation.generate(
        pipeline,
        pipeline_type,
        conditioning,
        pooled,
        Steps,
        Width,
        Height,
        Scale,
        Clip_Skip,
        generator,
        Inpainting_Strength,
        IP_Adapter,
        image_embeds,
        Inpainting_Image,
        Mask_Image,
        controlnets_scale,
        images,
        ref_image,
        Denoising_Strength
    )

    # Saving the image and resetting the output
    generated_image_savefile= image_saver.save_image(image, Prompt, prefix, base_path)
    clear_output()
    display(ui)

    # Saving the set parameters (second phase)
    save_param(f"{base_path}/Saved Parameters/main_parameters.json", dictionary)

    # Saving the last generated image's path
    last_generation_json = os.path.join(base_path, "last_generation.json")
    save_last(last_generation_json, generated_image_savefile, prefix)

    # Displaying the image
    display(image)
    print(f"Scheduler: {''.join(Scheduler_used)}")
    print(f"Seed: {generator_seed}")
    print(f"Image is saved at {generated_image_savefile}.")
    torch.cuda.empty_cache()
