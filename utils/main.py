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
import gc
import os

main = None

# Variables to avoid loading the same model or pipeline twice
class MainVar:
    def __init__(self):
        self.pipeline = None
        self.loaded_model = ""
        self.loaded_pipeline = ""
        self.vae_current = None
        self.loaded_controlnet_model = [None] * 3
        self.controlnets = [None] * 3
        self.images = [None] * 3
        self.controlnets_scale = [None] * 3

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

def controlnet_flush(main_class):
    cn_reset = ""
    cn_reset_sanitized_list = [element for element in main_class.loaded_controlnet_model if element]
    for weight in cn_reset_sanitized_list:
        if cn_reset_sanitized_list.index(weight) == (len(cn_reset_sanitized_list) - 1) and len(cn_reset_sanitized_list) > 1:
            cn_reset += f"and {weight} ControlNets"
        elif len(cn_reset_sanitized_list) == 1:
            cn_reset += f"{weight} ControlNet"
        else:
            cn_reset += f"{weight}, " if len(cn_reset_sanitized_list) == 3 else f"{weight} "
    print(f"You previously activated the {cn_reset} ControlNet. Because of this, the pipeline must be reloaded to free up some VRAM.")
    print("Flushing...")
    
    to_be_reset = [
        main_class.controlnets, 
        main_class.loaded_controlnet_model, 
        main_class.images, 
        main_class.controlnets_scale, 
        main_class.self.loaded_pipeline
    ]
    for value in to_be_reset:
        if isinstance(value, list):
            for element in value:
                element = None
        else:
            if value:
                del value
                value = None
    if main_class.pipeline:
        main_class.pipeline.to("cpu")
        del main_class.pipeline
        main_class.pipeline = None
    torch.cuda.empty_cache()
    gc.collect()

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

# Initializing image generation
def run(values_in_list, lora, embeddings, ip, hf_token, civit_token, ui, seed_list, dictionary, widgets_change, base_path):
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
        Canny_link, pipeline_type = controlnet_path_selector(Canny_Link, pipeline_type, base_path)
    else:
        Canny_link = ""

    if Depth_Map and selected_tab_for_pipeline == 2:
        Depthmap_Link, pipeline_type = controlnet_path_selector(DepthMap_Link, pipeline_type, base_path)
    else:
        Depthmap_Link = ""

    if Open_Pose and selected_tab_for_pipeline == 2:
        Openpose_Link, pipeline_type = controlnet_path_selector(OpenPose_Link, pipeline_type, base_path)
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

    # Instantiating the variables
    global main
    if not main:
        main = MainVar()

    # Flushing ControlNet model if deactivated after being used
    if (not Canny and main.controlnets[0]) or (not Depth_Map and main.controlnets[1]) or (not Open_Pose and main.controlnets[2]): 
        controlnet_flush(main)

    # Loading ControlNet
    if pipeline_type == "controlnet" and (Canny or Depth_Map or Open_Pose) and (Canny_link or Depthmap_Link or Openpose_Link):
        # Handling Canny
        if Canny and Canny_link is not None:
            if "canny" not in main.loaded_controlnet_model:
                print("Loading Canny...")
                main.loaded_controlnet_model[0] = "canny"
                main.controlnets[0] = ControlNetModel.from_pretrained(
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
            time.sleep(1)
            display(make_image_grid([c_img, canny_image.resize((1024, 1024))], rows=1, cols=2))
            main.images[0] = canny_image.resize((1024, 1024))
            main.controlnets_scale[0] = Canny_Strength
            
        # Handling Depth Map
        if Depth_Map and Depthmap_Link is not None:
            if "depth" not in main.loaded_controlnet_model:
                print("Loading Depth Map...")
                main.loaded_controlnet_model[1] = "depth"
                main.controlnets[1] = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    use_safetensors=True, 
                    low_cpu_mem_usage=True
                ).to("cuda")
            print("Converting image with Depth Map...")
            depth_estimator = pipe("depth-estimation", device="cpu")
            image_depth = Depthmap_Link.resize((1024, 1024))
            depth_map = get_depth_map(image_depth, depth_estimator).unsqueeze(0).half().to("cpu")
            main.images[1] = depth_map
            main.controlnets_scale[1] = Depth_Strength
            depth_map_display = ImagePIL.fromarray(get_depth_map_display(image_depth, depth_estimator))
            print("Depth Map is complete.")
            time.sleep(1)
            display(make_image_grid([image_depth, depth_map_display], rows=1, cols=2))
            
        # Handling Open Pose
        if Open_Pose and Openpose_Link is not None:
            if "openpose" not in main.loaded_controlnet_model:
                print("Loading Open Pose...")
                main.loaded_controlnet_model[2] = "openpose"
                main.controlnets[2] = ControlNetModel.from_pretrained(
                    "thibaud/controlnet-openpose-sdxl-1.0", 
                    torch_dtype=torch.float16, 
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                ).to("cuda")
            print("Converting image with Open Pose...")
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cpu")
            openpose_image = openpose(Openpose_Link)
            main.images[2] = openpose_image.resize((1024, 1024))
            main.controlnets_scale[2] = Open_Pose_Strength
            print("Open Pose is done.")
            display(make_image_grid([image_openpose, openpose_image.resize((1024, 1024))], rows=1, cols=2))
            
    # Handling pipeline and model loading
    main.pipeline, model_name = pipeline_selector.load_pipeline(
        main.pipeline,
        Model, 
        widgets_change[1], 
        main.loaded_model, 
        main.loaded_pipeline,
        pipeline_type,
        controlnets=main.controlnets, 
        active_inpaint=active_inpaint, 
        hf_token=HF_Token, 
        civit_token=Civit_Token,
        base_path=base_path
    )

    # Handling VAE
    if VAE_Link and (VAE_Link != main.vae_current or not main.vae_current):
        vae, loaded_vae = vae_loader.load_vae(
            main.vae_current, 
            VAE_Link, 
            VAE_Config, 
            widgets_change[0], 
            HF_Token, 
            Civit_Token,
            base_path=base_path
        )
        main.vae_current = loaded_vae
        if vae is not None:
            main.pipeline.vae = vae

    # Using a custom image encoder if IP-Adapter is True
    if IP_Adapter != "None":
        main.pipeline.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to("cuda")

    # Assigning new values 
    main.loaded_model = model_name
    main.loaded_pipeline = pipeline_type

    # Xformer, generator, and safety checker
    main.pipeline.enable_xformers_memory_efficient_attention()
    generator = torch.Generator("cpu").manual_seed(generator_seed)
    main.pipeline.safety_checker = None

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
        main.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM++ 2M SDE":
        main.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(main.pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", **scheduler_args)
    elif Scheduler == "DPM++ SDE":
        main.pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM2":
        main.pipeline.scheduler = KDPM2DiscreteScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DPM2 a":
        main.pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DDPM":
        main.pipeline.scheduler = DDPMScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Euler":
        main.pipeline.scheduler = EulerDiscreteScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Euler a":
        main.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "Heun":
        main.pipeline.scheduler = HeunDiscreteScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "LMS":
        main.pipeline.scheduler = LMSDiscreteScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DEIS":
        main.pipeline.scheduler = DEISMultistepScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "UniPC":
        main.pipeline.scheduler = UniPCMultistepScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "DDIM":
        main.pipeline.scheduler = DDIMScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)
    elif Scheduler == "PNDM":
        main.pipeline.scheduler = PNDMScheduler.from_config(main.pipeline.scheduler.config, **scheduler_args)

    # Using prompt weighting with Compel
    compel = Compel(tokenizer=[main.pipeline.tokenizer, main.pipeline.tokenizer_2], text_encoder=[main.pipeline.text_encoder, main.pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True], truncate_long_prompts=False)
    conditioning, pooled = compel([Prompt, Negative_Prompt])

    # Loading LoRA if not empty
    if LoRA_URLs:
        lora_loader.process(
            main.pipeline, 
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
            main.pipeline, 
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
        main.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=IP_Adapter, low_cpu_mem_usage=True)
        main.pipeline.set_ip_adapter_scale(IP_Adapter_Strength)
    torch.cuda.empty_cache()

    # Generating image
    prefix, image = run_generation.generate(
        main.pipeline,
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
