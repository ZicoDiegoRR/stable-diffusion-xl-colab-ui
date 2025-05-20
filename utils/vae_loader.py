from StableDiffusionXLColabUI.utils import downloader
from diffusers import AutoencoderKL
import os

def vae_url_checker(model_path):
    # Checking
    return model_path.startswith("https://") or model_path.startswith("http://")

def autoencoderkl_load(vae_path):
    # Loading the VAE based on whether it's a pretrained model from Hugging Face or not
    try:
        if vae_path[1]:
            return AutoencoderKL.from_single_file(vae_path[0], config=vae_path[1], torch_dtype=torch.float16, local_files_only=True)
        else:
            return AutoencoderKL.from_pretrained(vae_path[0], torch_dtype=torch.float16)
    except Exception as e:
        print(f"Error when loading the VAE. Reason: {e}")
        print("Skipped VAE.")
        return None


def load_vae(current_vae, model_path, config_path, hf_token, civit_token):
    # Checking if the provided vae has been loaded
    if model_path != current_vae:
        # Determining the path, whether the VAE has been downloaded or not
        if vae_url_checker(model_path):
            vae_path = [downloader.download_file(model_path, "VAE", hf_token, civit_token), downloader.download_file(config_path, "VAE", hf_token, civit_token)]
            vae_save_folder = os.path.splitext(os.path.basename(vae_path))
            os.makedirs(f"/content/VAE/{vae_save_folder}", exist_ok=True)
            for path in vae_path:
                vae_filename = os.path.basename(path)
                os.rename(path, f"/content/VAE/{vae_save_folder}/{vae_filename}")

        # For Hugging Face pretrained VAE models
        elif model_path.count("/") == 1:
            vae_path = [model_path, None]

        # For VAE from local files
        else:
            vae_path = os.listdir(f"/content/VAE/{model_path}") if not model_path.startswith("/content/VAE") else os.listdir(model_path)

        # Load
        vae = autoencoderkl_load(vae_path)
        
    # Skipping the VAE if the model is still in an URL form, but the config is empty
    elif vae_url_checker(model_path) and not config_path:
        print("You inputted a link to the VAE model, but not the config file. It's mandatory to pass both links.")
        print("Skipped VAE.")
        vae = None

    return vae
        
