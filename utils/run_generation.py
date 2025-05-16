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
    Inpainting_Strength,
    image_embeds=None,
    inpaint_image=None,
    mask_image=None,
    controlnets_scale=None,
    images=None,
    ref_image=None,
    Denoising_Strength=None
):
    if pipeline_type == "text2img": # For Text2Img
        image_prefix = "[Text-to-Image]"
        if IP_Adapter == "None":
            image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=Steps,
                width=Width,
                height=Height,
                guidance_scale=Scale,
                clip_skip=Clip_Skip,
                generator=generator
            ).images[0]
        else:
            image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=Steps,
                ip_adapter_image=image_embeds,
                width=Width,
                height=Height,
                guidance_scale=Scale,
                clip_skip=Clip_Skip,
                generator=generator
            ).images[0]
            pipeline.unload_ip_adapter()
    elif pipeline_type == "inpaint": # For Inpainting
        image_prefix = "[Inpainting]"
        if IP_Adapter == "None":
            image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=Steps,
                width=Width,
                height=Height,
                guidance_scale=Scale,
                clip_skip=Clip_Skip,
                image=inpaint_image,
                mask_image=mask_image,
                generator=generator,
                strength=Inpainting_Strength
            ).images[0]
        else:
            image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=Steps,
                ip_adapter_image=image_embeds,
                width=Width,
                height=Height,
                guidance_scale=Scale,
                clip_skip=Clip_Skip,
                generator=generator,
                image=inpaint_image,
                mask_image=mask_image,
                strength=Inpainting_Strength
            ).images[0]
            pipeline.unload_ip_adapter()
    elif pipeline_type == "controlnet": # For ControlNet
        image_prefix = "[ControlNet]"
        if IP_Adapter == "None":
          image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                clip_skip=Clip_Skip,
                num_inference_steps=Steps,
                generator=generator,
                width=Width,
                height=Height,
                image=[element for element in images if element is not None],
                controlnet_conditioning_scale=[element for element in controlnets_scale if element],
                guidance_scale=Scale
          ).images[0]
        else:
          image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=Steps,
                ip_adapter_image=image_embeds,
                width=Width,
                height=Height,
                guidance_scale=Scale,
                clip_skip=Clip_Skip,
                generator=generator,
                image=[element for element in images if element is not None],
                controlnet_conditioning_scale=[element for element in controlnets_scale if element]
          ).images[0]
    else: # For Img2img
        image_prefix = "[Image-to-Image]"
        if IP_Adapter == "None":
            image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=Steps,
                width=Width,
                height=Height,
                image=ref_image,
                strength=Denoising_Strength,
                guidance_scale=Scale,
                clip_skip=Clip_Skip,
                generator=generator
            ).images[0]
        else:
            image = pipeline(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=Steps,
                ip_adapter_image=image_embeds,
                image=ref_image,
                strength=Denoising_Strength,
                width=Width,
                height=Height,
                guidance_scale=Scale,
                clip_skip=Clip_Skip,
                generator=generator
            ).images[0]
            pipeline.unload_ip_adapter()

    return image_prefix
