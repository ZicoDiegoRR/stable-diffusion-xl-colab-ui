from StableDiffusionXLColabUI.utils import downloader
from diffusers import AutoencoderKL
import torch
import os

def vae_url_checker(model_path):
    # Checking
    return model_path.startswith("https://") or model_path.startswith("http://")

def autoencoderkl_load(vae_path):
    # Loading the VAE based on whether it's a pretrained model from Hugging Face or not
    try:
        if vae_path[1]:
            return AutoencoderKL.from_single_file(vae_path[0], config=vae_path[1], torch_dtype=torch.float16, local_files_only=True).to("cuda")
        else:
            return AutoencoderKL.from_pretrained(vae_path[0], torch_dtype=torch.float16).to("cuda")
    except Exception as e:
        print(f"Error when loading the VAE model from {vae_path[0]}. Skipped VAE.")
        print(f"Reason: {e}")
        return None

def post_download(vae_download_path):
    vae_path = []
    vae_save_folder, _ = os.path.splitext(os.path.basename(vae_download_path[0]))
    os.makedirs(f"/content/VAE/{vae_save_folder}", exist_ok=True)
    for path in vae_download_path:
        vae_filename = os.path.basename(path)
        vae_destination = f"/content/VAE/{vae_save_folder}/{vae_filename}"
                
        os.rename(path, vae_destination)
        vae_path.append(vae_destination)
    return vae_path

def download_vae(model_path, type, hf_token, civit_token, base_path, config=None):
    vae_weight_download = downloader.download_file(
        model_path, 
        "VAE", 
        hf_token, 
        civit_token,
        base_path=base_path
    )
    vae_weight_name, _ = os.path.splitext(os.path.basename(vae_weight_download)) 
    vae_config_download = downloader.download_file(
        config if config and (config.startswith("https://") or config.startswith("http://") or config.startswith("/content/")) else vae_weight_name, 
        "VAE",
        hf_token, 
        civit_token,
        base_path=base_path,
        subfolder=vae_weight_name
    )
    
    vae_path = post_download([
        vae_weight_download, 
        vae_config_download
    ])
    return vae_path

def load_vae(current_vae, model_path, config_path, widget, hf_token, civit_token, base_path):
    os.makedirs("/content/VAE", exist_ok=True)
    
    # Checking if the provided vae has been loaded
    if model_path != current_vae:
        # Determining the path, whether the VAE has been downloaded or not
        if vae_url_checker(model_path):
            vae_path = download_vae(
                model_path, 
                type, 
                hf_token, 
                civit_token, 
                base_path, 
                config=config_path, 
            )
            for file in vae_path:
                vae_filename = os.path.basename(file)
                widget_value, _ = os.path.splitext(vae_filename)
                widget[i].value = widget_value
            for_vae_current = os.path.splitext(os.path.basename(vae_path[0])) 

        # For Hugging Face pretrained VAE models
        elif model_path.count("/") == 1:
            vae_path = [model_path, None]
            for_vae_current = model_path

        # For VAE from local files
        else:
            if not model_path.startswith("/content/VAE"):
                if os.path.exists(f"/content/VAE/{model_path}"):
                    vae_path_collected = [os.path.join(f"/content/VAE/{model_path}", path) for path in os.listdir(f"/content/VAE/{model_path}") if os.path.isfile(os.path.join(f"/content/VAE/{model_path}", path))]
                else:
                    vae_path_collected = ["", ""]
            else: 
                vae_subfolder, _ = os.path.splitext(os.path.basename(model_path)) 
                vae_path_collected = [os.path.join(f"/content/VAE/{vae_subfolder}", path) for path in os.listdir(f"/content/VAE/{vae_subfolder}") if os.path.isfile(os.path.join(f"/content/VAE/{vae_subfolder}", path))]

            vae_path_first = ["", ""]
            for element in vae_path_collected:
                if element.endswith(".json"):
                    vae_path_first[1] = element
                else:
                    vae_path_first[0] = element

            if not vae_path_first[0]:
                vae_path = download_vae(
                    model_path, 
                    type, 
                    hf_token, 
                    civit_token, 
                    base_path, 
                    config=config_path, 
               )
            else:
                vae_path = vae_path_first
            for_vae_current = os.path.splitext(os.path.basename(vae_path[0])) 

        # Load
        vae = autoencoderkl_load(vae_path)

        # Add new value to the vae_current
        loaded_vae, _ = os.path.splitext(os.path.basename(vae_path[0])) 
    
    # Skipping the VAE if the model is still in an URL form, but the config is empty
    elif vae_url_checker(model_path) and not config_path:
        print("You inputted a link to the VAE model, but not the config file. It's mandatory to pass both links.")
        print("Skipped VAE.")
        vae = None
        loaded_vae = current_vae

    return vae, loaded_vae
        
