import json
import os

def load_param(filename):
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        return []
        
def old_to_new(path):
    default_param_for_dict = {
        "text2img": ["", "", "", 1024, 1024, 12, 6, 2, "Default (defaulting to the model)", 
                     False, False, False, False, "", "",
                    ],
        "img2img": ["", "", "", 1024, 1024, 12, 6, 2, "Default (defaulting to the model)", 
                    False, False, False, False, "", "", "", 0.3,
                   ],
        "controlnet": ["", "", "", 1024, 1024, 12, 6, 2, "Default (defaulting to the model)", 
                      False, False, False, False, "", "", "", 100, 240, False, 
                      0.7, "", False, 0.7, "", False, 0.7,
                      ],
        "inpaint": ["pre-generated text2image image", "", False, 0.9],
        "ip": ["", 0.8, "None"],
        "lora": ["", ""],
        "embeddings": ["", ""],
    }
    
    target_list = load_param(path)
    if len(target_list) < 41:
        for i in range(41 - len(target_list)):
            target_list.append(None)
            
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
    
    img2img_list = [
        target_list[37],
        target_list[38],
    ]
    
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
    
    inpaint_list = [
        target_list[24],
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
        target_list,
    ]
