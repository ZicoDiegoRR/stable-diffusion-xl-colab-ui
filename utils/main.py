from StableDiffusionXLColabUI.utils import (
    embeddings_loader, 
    image_saver,
    lora_loader,
    pipeline_selector,
    run_generation,
    vae_loader,
)

def run(values_in_list, hf_token, civit_token, ui, seed_and_bool):
    pipeline_type = ""
    if len(values_in_list) == 15:
        pipelie_type = "text2img"
    elif len(values_in_list) == 17:
        pipelie_type = "img2img"
    if len(values_in_list) == 26:
        pipelie_type = "controlnet"

    
