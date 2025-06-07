import json
import os

# Removing None values
def sanitize_none_values(list_value, default_value):
    sanitized_list = []
    for i, value in enumerate(list_value):
        if value is None:
            sanitized_list.append(default_value[i])
        else:
            sanitized_list.append(value)
    return sanitized_list

# Converting the save file
def old_to_new(cfg):
    default_param_for_dict = [
        "",
        "",
        "Safe Tensor (.safetensors)",
        "",
        1024,
        1024,
        "Default (defaulting to the model)",
        12,
        6,
        "",
        2,
        "",
        "",
        100,
        240,
        "",
        False,
        0.7,
        "",
        False,
        0.7,
        "",
        False,
        0.7,
        "pre-generated text2image image",
        "",
        False,
        0.9,
        "None",
        "",
        0.7,
        False,
        False,
        False,
        False,
        False,
        "",
        "",
        0.3,
        "",
        "",
    ]

    # Loading and checking if the list contains less than 41 variables
    if len(cfg) < 41:
        for i in range(41 - len(cfg)):
            cfg.append(None)

    # Sanitized list
    target_list = sanitize_none_values(cfg, default_param_for_dict)

    # Text2Img or general parameters
    text2img_or_general_list = [
        target_list[0],
        target_list[3],
        target_list[1],
        target_list[4],
        target_list[5],
        target_list[7],
        target_list[8],
        target_list[10],
        target_list[6],
        target_list[32],
        target_list[33],
        target_list[34],
        target_list[35],
        target_list[9],
        target_list[36],
    ]

    # Img2Img-exclusive parameters
    img2img_list = [
        target_list[37],
        target_list[38],
    ]

    # ControlNet-exclusive parameters
    controlnet_list = [
        target_list[15],
        target_list[13],
        target_list[14],
        target_list[16],
        target_list[17],
        target_list[18],
        target_list[19],
        target_list[20],
        target_list[21],
        target_list[22],
        target_list[23],
    ]

    # Inpainting-exclusive parameters
    inpaint_list = [
        target_list[24],
        target_list[25],
        target_list[26],
        target_list[27],
    ]

    # IP-Adapter-exclusive parameters
    ip_list = [
        target_list[29],
        target_list[30],
        target_list[28],
    ]

    # LoRA-exclusive parameters
    lora_list = [
        target_list[11],
        target_list[12],
    ]
    # Textual Inversion-exclusive parameters
    ti_list = [
        target_list[39],
        target_list[40],
    ]

    # Unused in the new version (for selecting the model format and saving the freeze widget's value)
    ''' 
    misc_list = [
        target_list[31],
        target_list[2],
    ]
    '''

    # Converting list to dictionary
    new_cfg = { # Since the old save uses list and each pipeline shares similar parameters, 
                # there's nothing I can do but to share the general parameters across each pipeline
        "text2img": text2img_or_general_list,
        "img2img": text2img_or_general_list + img2img_list,
        "controlnet": text2img_or_general_list + controlnet_list,
        "inpaint": inpaint_list,
        "ip": ip_list,
        "lora": lora_list,
        "embeddings": ti_list,
    }
    return new_cfg
