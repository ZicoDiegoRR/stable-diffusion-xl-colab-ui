import os
import gc
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLControlNetPipeline,  
)
from StableDiffusionXLColabUI.utils import downloader

def raise_error(Model_path):
    if not os.path.exists(Model_path):
        Error = f"Model {Model_path} doesn't exist."
        Warning = ""
    else:
        Error = f"The model {Model_path} contains unsupported file or the download was corrupted. "
        if not civit_token and "civitai.com" in Model:
            Warning = "You inputted a CivitAI's link, but your token is empty. It's possible that you got unauthorized access during the download."
        elif "huggingface.co" in Model or Model.count("/") == 1:
            if not hf_token:
                Token_Error = "but the Hugging Face's token is empty. Are you trying to access a private model or the repository doesnt have model_index.json?"
            else:
                Token_Error = "but the model couldn't be loaded properly. Make sure it's the correct model and you paste the URL from 'Copy download link' option."
            Warning = f"You tried to access the model from Hugging Face, {Token_Error}"
        else:
            Warning = "Did you input the correct link or path? Or did you use the correct format?"
    if os.path.exists(Model_path):
        os.remove(Model_path)
    raise TypeError(f"Program quit with an error: {Error}{Warning}")

def flush(pipe, new, old, type):
    if type == "model":
        warning_restart = f"You inputted a new model ({new}), which is different than the old one ({old})."
    elif type == "pipeline":
        warning_restart = f"You changed the pipeline from {old} to {new}."
    print(f"{warning_restart} Flushing the model...")
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

def verify(pipe, model_url, loaded_model, loaded_pipeline, pipeline_type):
    if loaded_model:
        if loaded_model != model_url:
            flush(pipe, model_url, loaded_model, "model")
            model_check = True
        else:
            model_check = False
    else:
        model_check = True
        
    if loaded_pipeline: 
        if loaded_pipeline != pipeline_type:
            flush(pipe, pipeline_type, loaded_pipeline, "pipeline")
            pipe_check = True
        else:
            pipe_check = False
    else:
        pipe_check = True
        
    if model_check or pipe_check:
        return True
    elif not model_check and not pipe_check:
        return False

def load_pipeline(pipe, model_url, widget, loaded_model, loaded_pipeline, pipeline_type, format="safetensors", controlnets=None, active_inpaint=False, vae=None, hf_token="", civit_token="", base_path="/content"):
    # For Hugging Face repository with "author/repo_name" format
    if model_url.count("/") == 1 and (not model_url.startswith("https://") or not model_url.startswith("http://")):
        if verify(pipe, model_url, loaded_model, loaded_pipeline, pipeline_type) or pipe is None:
            try:
                if pipeline_type == "text2img" and not active_inpaint:
                    pipeline = StableDiffusionXLPipeline.from_pretrained(model_url, torch_dtype=torch.float16).to("cuda")
                elif active_inpaint and pipeline_type == "inpaint":
                    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(model_url, torch_dtype=torch.float16).to("cuda")
                elif pipeline_type == "controlnet":
                    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(model_url, controlnet=None, torch_dtype=torch.float16).to("cuda")
                elif pipeline_type == "img2img":
                    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_url, torch_dtype=torch.float16).to("cuda")
                model_name = model_url
            except (ValueError, OSError):
                raise_error(model_url)
        else:
            return pipe, model_url

    # For non-Hugging Face repository or Hugging Face direct link
    else:
        os.makedirs("/content/Checkpoint", exist_ok=True)
        
        # Download
        if model_url.startswith("https://") or model_url.startswith("http://"):
            Model_path = downloader.download_file(model_url, "Checkpoint", hf_token, civit_token, base_path)
            model_name, _ = os.path.splitext(os.path.basename(Model_path))
            widget.value = model_name
        else:
            if not model_url.startswith("/content/Checkpoint"):
                Model_path = downloader.download_file(model_url, "Checkpoint", hf_token, civit_token, base_path)
                model_name, _ = os.path.splitext(os.path.basename(Model_path))
            else:
                Model_path = model_url
                model_name = model_url
                
        # Load
        if verify(pipe, model_url, loaded_model, loaded_pipeline, pipeline_type) or pipe is None:
            try:
                if pipeline_type == "text2img" and not active_inpaint:
                    pipeline = StableDiffusionXLPipeline.from_single_file(Model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
                elif active_inpaint and pipeline_type == "inpaint":
                    pipeline = StableDiffusionXLInpaintPipeline.from_single_file(Model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
                elif pipeline_type == "controlnet":
                    pipeline = StableDiffusionXLControlNetPipeline.from_single_file(Model_path, controlnet=None torch_dtype=torch.float16, variant="fp16").to("cuda")
                elif pipeline_type == "img2img":
                    pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(Model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
                
        # Raise an  error if there's something wrong when loading the model
            except (ValueError, OSError):
                raise_error(Model_path)
        else:
            return pipe, model_url

    return pipeline, model_name
