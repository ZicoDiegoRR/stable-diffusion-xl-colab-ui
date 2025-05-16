from StableDiffusionXLColabUI.utils import downloader
from diffusers import AutoencoderKL
import os

def load_vae(model_path, config_path, hf_token, civit_token):
    # Checking if the provided vae has been loaded
    global current_vae
    if model_path != current_vae:
        # Determining the path, whether the VAE has been downloaded or not
        if model_path.startswith("https://") or model_path.startswith("http://"):
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

        # Loading the VAE based on whether it's a pretrained model from Hugging Face or not
        if vae_path[1]:
            try:
                vae = AutoencoderKL.from_single_file(vae_path[0], config=vae_path[1], torch_dtype=torch.float16, local_files_only=True)
                current_vae = model_path
            except Exception as e:
                print(f"Error when loading the VAE. Reason: {e}")
                print("Skipped VAE.")
                vae = None
        else:
            try:
                vae = AutoencoderKL.from_pretrained(vae_path[0], torch_dtype=torch.float16)
                current_vae = model_path
            except Exception as e:
                print(f"Error when loading the VAE. Reason: {e}")
                print("Skipped VAE.")
                vae = None

    # Skipping the VAE if the model is still in an URL form, but the config is empty
    elif (model_path.startswith("https://") or model_path.startswith("http://")) and not config_path:
        print("You inputted a link to the VAE model, but not the config file. It's mandatory to pass both links.")
        print("Skipped VAE.")
        vae = None
        
