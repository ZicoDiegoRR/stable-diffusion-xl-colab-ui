import os
from diffusers import (
    ControlNetModel, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLControlNetPipeline, 
    StableDiffusionXLInpaintPipeline, 
    AutoencoderKL,
)
from StableDiffusionXLColabUI.utils import downloader

def load_pipeline(model_url, format, controlnets, active_inpaint, pipeline_type, loaded_model, hf_token, civit_token):
    Model = model_url
    if Model.count("/") == 1 and (not Model.startswith("https://") or not Model.startswith("http://")):
        if all(cn is None for cn in controlnets) and pipeline_type != "img2img" and not active_inpaint and (pipeline_type != "text2img" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
            if VAE_Link:
                pipeline = StableDiffusionXLPipeline.from_pretrained(Model, vae=vae, torch_dtype=torch.float16).to("cuda")
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(Model, torch_dtype=torch.float16).to("cuda")
        elif active_inpaint and pipeline_type != "img2img" and all(cn is None for cn in controlnets) and (pipeline_type != "inpaint" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
            if VAE_Link:
                pipeline = AutoPipelineForInpainting.from_pretrained(Model, vae=vae, torch_dtype=torch.float16).to("cuda")
            else:
                pipeline = AutoPipelineForInpainting.from_pretrained(Model, torch_dtype=torch.float16).to("cuda")
        elif pipeline_type != "img2img" and (pipeline_type != "controlnet" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
            if VAE_Link:
                pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(Model, controlnet=[element for element in controlnets if element], vae=vae, torch_dtype=torch.float16).to("cuda")
            else:
                pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(Model, controlnet=[element for element in controlnets if element], torch_dtype=torch.float16).to("cuda")
        elif pipeline_type == "img2img" and (pipeline_type != "img2img" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
            if VAE_Link:
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(Model, vae=vae, torch_dtype=torch.float16).to("cuda")
            else:
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(Model, torch_dtype=torch.float16).to("cuda")
    else:
        if Model.startswith("https://") or Model.startswith("http://"):
            Model_path = downloader.download_file(Model, "Checkpoint", hf_token, civit_token)
        else:
            Model_path = f"/content/Checkpoint/{Model}.{format}"
        try:
            if all(cn is None for cn in controlnets) and pipeline_type != "img2img" and not active_inpaint and (pipeline_type != "text2img" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
                if VAE_Link:
                    pipeline = StableDiffusionXLPipeline.from_single_file(Model_path, vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
                else:
                    pipeline = StableDiffusionXLPipeline.from_single_file(Model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
            elif pipeline_type != "img2img" and active_inpaint and all(cn is None for cn in controlnets) and (pipeline_type != "inpaint" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
                if VAE_Link:
                    pipeline = AutoPipelineForInpainting.from_single_file(Model_path, vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
                else:
                    pipeline = AutoPipelineForInpainting.from_single_file(Model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
            elif pipeline_type != "img2img" and (pipeline_type != "controlnet" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
                if VAE_Link:
                    pipeline = StableDiffusionXLControlNetPipeline.from_single_file(Model_path, controlnet=[element for element in controlnets if element], vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
                else:
                    pipeline = StableDiffusionXLControlNetPipeline.from_single_file(Model_path, controlnet=[element for element in controlnets if element], torch_dtype=torch.float16, variant="fp16").to("cuda")
            elif pipeline_type == "img2img" and (pipeline_type != "img2img" or not loaded_pipeline) and (not loaded_model or model_widget.value != loaded_model):
                if VAE_Link:
                    pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(Model_path, vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
                else:
                    pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(Model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
        except (ValueError, OSError):
            pass
            if not os.path.exists(Model_path):
                os.remove(Model_path)
            if not civit_token and "civitai.com" in Model:
                Warning = "You inputted a CivitAI's link, but your token is empty. It's possible that you got unauthorized access during the download."
            elif "huggingface.co" in Model or Model.count("/") == 1:
                if not hf_token:
                    Token_Error = "but the Hugging Face's token is empty. Are you trying to access a private model or the repository doesnt have model_index.json?"
                else:
                    Token_Error = "but the model couldn't be loaded properly. Make sure it's the correct model and you paste the URL from 'Copy download link' option."
                Warning = f"You tried to access the model from Hugging Face, {Token_Error}"
            else:
                Warning = "Did you input the correct link? Or did you use the correct format?"
            raise TypeError(f"The link ({Model}) contains unsupported file or the download was corrupted. {Warning}")
