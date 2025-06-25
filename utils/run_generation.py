from PIL import Image

def generate(
    pipeline,
    pipeline_type,
    conditioning,
    pooled,
    Steps,
    Width,
    Height,
    Scale,
    Clip_Skip,
    generator,
    Inpainting_Strength=None,
    IP_Adapter="None",
    image_embeds=None,
    inpaint_image=None,
    mask_image=None,
    controlnet_modes=None,
    controlnets_scale=None,
    images=None,
    ref_image=None,
    Denoising_Strength=None
):
    # Main arguments
    generation_arguments = {
        "prompt_embeds": conditioning[0:1],
        "pooled_prompt_embeds": pooled[0:1],
        "negative_prompt_embeds": conditioning[1:2],
        "negative_pooled_prompt_embeds": pooled[1:2],
        "num_inference_steps": Steps,
        "width": Width,
        "height": Height,
        "guidance_scale": Scale,
        "clip_skip": Clip_Skip,
        "generator": generator
    }

    # Argument validation based on the pipeline and adapter
    if IP_Adapter != "None": # If IP-Adapter is turned on
        generation_arguments["ip_adapter_image"] = image_embeds
        
    if pipeline_type == "text2img": # For Text2Img
        image_prefix = "[Text-to-Image]"

    elif pipeline_type == "inpaint": # For Inpainting
        image_prefix = "[Inpainting]"
        generation_arguments["image"] = Image.open(inpaint_image)
        generation_arguments["mask_image"] = Image.open(mask_image)
        generation_arguments["strength"] = Inpainting_Strength

    elif pipeline_type == "controlnet": # For ControlNet
        image_prefix = "[ControlNet]"
        generation_arguments["control_image"] = [element for element in images if element is not None]
        generation_arguments["control_mode"] = [element for element in controlnet_modes if element is not None]
        generation_arguments["controlnet_conditioning_scale"] = [element for element in controlnets_scale if element is not None]

    else: # For Img2img
        image_prefix = "[Image-to-Image]"
        generation_arguments["image"] = ref_image
        generation_arguments["strength"] = Denoising_Strength

    image = pipeline(**generation_arguments).images[0]
    if IP_Adapter != "None": # If IP-Adapter is turned on
        pipeline.unload_ip_adapter()

    return image_prefix, image
