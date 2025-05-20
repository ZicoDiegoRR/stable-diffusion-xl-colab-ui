from StableDiffusionXLColabUI.utils import (
    embeddings_loader, 
    image_saver,
    lora_loader,
    pipeline_selector,
    run_generation,
    vae_loader,
)
import ipywidgets as widgets
import random

def run(values_in_list, hf_token, civit_token, ui, seed_list):
    # Initialization
    pipeline_type = ""
    if len(values_in_list) == 15:
        pipelie_type = "text2img"
    elif len(values_in_list) == 17:
        pipelie_type = "img2img"
    if len(values_in_list) == 26:
        pipelie_type = "controlnet"

    if not seed_list[1] or seed_list[0].value == -1:
        generator_seed = random.randint(1, 1000000000000)
    else:
        generator_seed = seed_list[0].value

    # Gathering values
    Prompt = values_in_list
    Negative_Prompt = values_in_list
    Model = values_in_list

    Width = values_in_list
    Height = values_in_list
    Steps = values_in_list
    Scale = values_in_list
    Clip_Skip = values_in_list

    Scheduler = values_in_list
    Karras = values_in_list
    V_Prediction = values_in_list
    SGMUniform = values_in_list
    Rescale_betas_to_zero_SNR = values_in_list
    
    VAE_Link = values_in_list
    VAE_Config = values_in_list
