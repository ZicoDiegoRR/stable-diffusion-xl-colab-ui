from StableDiffusionXLColabUI.utils import (
    embeddings_loader, 
    image_saver,
    lora_loader,
    pipeline_selector,
    run_generation,
    vae_loader,
    scheduler_selector,
    ip_adapter_loader,
    controlnet_loader,
)
from compel import Compel, ReturnedEmbeddingsType
from diffusers.utils import load_image, make_image_grid
from IPython.display import display, clear_output
from huggingface_hub import login
import ipywidgets as widgets
import random
import torch
import json
import time
import gc
import os

# Global variable
main = None

# Variables to avoid loading the same model or pipeline twice
class MainVar:
    def __init__(self):
        self.pipeline = None
        self.vae_current = None
        self.controlnet = None
        self.images = [None] * 3
        self.controlnets_scale = [None] * 3
        self.controlnet_modes = [None] * 3

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

# Initializing image generation
def run(values_in_list, lora, embeddings, ip, hf_token, civit_token, ui, seed_list, dictionary, widgets_change, base_path, get_image_class):
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

    # VARIABLES
    #____________________________________________________________________________________________________________________________________________________________________________
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
    #____________________________________________________________________________________________________________________________________________________________________________

    # PREPROCESS
    #____________________________________________________________________________________________________________________________________________________________________________
    # Logging in to HF hub if Hugging Face's token is not empty
    if hf_token:
      login(hf_token)

    # Selecting image and pipeline
    Canny_link = ""
    Depthmap_Link = ""
    Openpose_Link = ""
    if selected_tab_for_pipeline == 2:
        if Canny:
            Canny_link, pipeline_type = controlnet_loader.controlnet_path_selector(Canny_Link, pipeline_type, base_path)
        if Depth_Map:
            Depthmap_Link, pipeline_type = controlnet_loader.controlnet_path_selector(DepthMap_Link, pipeline_type, base_path)
        if Open_Pose:
            Openpose_Link, pipeline_type = controlnet_loader.controlnet_path_selector(OpenPose_Link, pipeline_type, base_path)

    active_inpaint = False
    if Inpainting and selected_tab_for_pipeline == 3:        
        if not Mask_Image:
            print("You checked Inpainting while you're leaving mask image empty. Mask image is required for Inpainting.")
            print("Skipped Inpainting.")
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
    #____________________________________________________________________________________________________________________________________________________________________________

    # RUNNING
    #____________________________________________________________________________________________________________________________________________________________________________
    
    # Handling ControlNet
    main.controlnet, main.images, main.controlnets_scale, main.controlnet_modes = controlnet_loader.load(
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
        main.controlnet,
        main.images,
        main.controlnets_scale,
        main.controlnet_modes,
        get_image_class,
    )
    
    # Handling pipeline and model loading
    main.pipeline, used_pipeline = pipeline_selector.load_pipeline(
        main.pipeline,
        Model, 
        widgets_change[1], 
        pipeline_type,
        active_inpaint=active_inpaint, 
        controlnets=main.controlnet,
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

    # Xformer, generator, and safety checker
    main.pipeline.enable_xformers_memory_efficient_attention()
    generator = torch.Generator("cpu").manual_seed(generator_seed)
    main.pipeline.safety_checker = None

    # Handling schedulers
    Scheduler_used = scheduler_selector.scheduler(
        main.pipeline,
        V_Prediction,
        Karras,
        Rescale_betas_to_zero_SNR,
        SGMUniform,
        Scheduler,
    )

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

    # Handling IP-Adapter
    image_embeds = None
    if IP_Adapter != "None" and IP_Image_Link:
        image_embeds = ip_adapter_loader.load(
            main.pipeline,
            IP_Adapter,
            IP_Image_Link,
            IP_Adapter_Strength,
        )
        
    # Generating image
    prefix, image = run_generation.generate(
        used_pipeline,
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
        main.controlnet_modes,
        main.controlnets_scale,
        main.images,
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
    last_generation_json = os.path.join(f"{base_path}/Saved Parameters", "last_generation.json")
    save_last(last_generation_json, generated_image_savefile, prefix)

    # Displaying the image
    display(image)
    print(f"Scheduler: {''.join(Scheduler_used)}")
    print(f"Seed: {generator_seed}")
    print(f"Image is saved at {generated_image_savefile}.")
    torch.cuda.empty_cache()
    gc.collect()
    #____________________________________________________________________________________________________________________________________________________________________________
