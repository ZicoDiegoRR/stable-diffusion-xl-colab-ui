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

def load_pipeline(pipe, model_url, widget, pipeline_type, format="safetensors", controlnets=None, active_inpaint=False, vae=None, hf_token="", civit_token="", base_path="/content"):
    # Download or input the URL or path to urls.json
    if model_url.startswith(("https://", "http://")) or model_url.count("/"):
        Model_path = downloader.download_file(model_url, "Checkpoint", hf_token, civit_token, base_path)
    else:
        Model_path = url
    
    # For Hugging Face repository with "author/repo_name" format
    if model_url.count("/") == 1 and (not model_url.startswith("https://") or not model_url.startswith("http://")):
        try:
            if not pipe:
                pipeline = StableDiffusionXLPipeline.from_pretrained(model_url, torch_dtype=torch.float16).to("cuda")
            else: 
                pipeline = pipe
                
            if pipeline_type == "text2img" and not active_inpaint and not isinstance(pipeline, StableDiffusionXLPipeline):
                pipeline = StableDiffusionXLPipeline(**pipeline.components).to("cuda")
            elif active_inpaint and pipeline_type == "inpaint" and not isinstance(pipeline, StableDiffusionXLInpaintPipeline):
                pipeline = StableDiffusionXLInpaintPipeline(**pipeline.components).to("cuda")
            elif pipeline_type == "controlnet" and not isinstance(pipeline, StableDiffusionXLControlNetPipeline):
                pipeline = StableDiffusionXLControlNetPipeline(**pipeline.components, controlnet=None).to("cuda")
            elif pipeline_type == "img2img" and  not isinstance(pipeline, StableDiffusionXLImg2ImgPipeline:
                pipeline = StableDiffusionXLImg2ImgPipeline(**pipeline.components).to("cuda")
        except (ValueError, OSError):
            raise_error(model_url)

    # For non-Hugging Face repository or Hugging Face direct link
    else:
        os.makedirs("/content/Checkpoint", exist_ok=True)
        widget.value, _ = os.path.splitext(os.path.basename(Model_path))
                
        # Load
        try:
            if not pipe:
                pipeline = StableDiffusionXLPipeline.from_single_file(Model_path, torch_dtype=torch.float16).to("cuda")
            else: 
                pipeline = pipe
                
            if pipeline_type == "text2img" and not active_inpaint and not isinstance(pipeline, StableDiffusionXLPipeline):
                pipeline = StableDiffusionXLPipeline(**pipeline.components).to("cuda")
            elif active_inpaint and pipeline_type == "inpaint" and not isinstance(pipeline, StableDiffusionXLInpaintPipeline):
                pipeline = StableDiffusionXLInpaintPipeline(**pipeline.components).to("cuda")
            elif pipeline_type == "controlnet" and not isinstance(pipeline, StableDiffusionXLControlNetPipeline):
                pipeline = StableDiffusionXLControlNetPipeline(**pipeline.components, controlnet=None).to("cuda")
            elif pipeline_type == "img2img" and  not isinstance(pipeline, StableDiffusionXLImg2ImgPipeline:
                pipeline = StableDiffusionXLImg2ImgPipeline(**pipeline.components).to("cuda")
        except (ValueError, OSError):
            raise_error(model_url)

    return pipeline
