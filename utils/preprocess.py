from StableDiffusionXLColabUI.utils import save_file_converter
from transformers import pipeline as pipe, set_seed
from google.colab import drive
import requests
import time
import json
import os

# Load the save file
def load_param(filename):
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        return []

#Function to save parameters config (had to make separate JSON def to avoid confusion)
def save_param(path, data):
    with open(path, 'w') as file:
        json.dump(data, file)

# Return the default values for the parameters
def default_params():
  return {
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

# Checking and validating the save file
def list_or_dict(cfg):
  if isinstance(cfg, list):
    new_cfg = save_file_converter.old_to_new(cfg)
    return new_cfg
  elif isinstance(cfg, dict):
    return cfg
  else:
    new_cfg = default_params()
    return new_cfg

def run():
    # Loading the saved config for the IPyWidgets
    if not os.path.exists("/content/gdrive/MyDrive"):
        try:
            drive.mount('/content/gdrive', force_remount=True)
        except Exception as e:
            print("Excluding Google Drive storage...")
            time.sleep(1.5)

    if os.path.exists("/content/gdrive/MyDrive"):
        base_path = "/content/gdrive/MyDrive"
    else:
        base_path = "/content"

    # Loading the save file
    cfg = None
    if os.path.exists(f"{base_path}/parameters.json"):
        cfg = load_param(os.path.join(f"{base_path}", "parameters.json"))
        print(f"Found a config at {base_path}/parameters.json.")
        os.makedirs(f"{base_path}/Saved Parameters", exist_ok=True)
        save_param(os.path.join(f"{base_path}/Saved Parameters/", "main_parameters.json"), cfg)
    elif not os.path.exists(f"{base_path}/parameters.json") or os.path.exists(f"{base_path}/Saved Parameters/main_parameters.json"):
        cfg = load_param(os.path.join(f"{base_path}/Saved Parameters/", "main_parameters.json"))
        print(f"Found a config at {base_path}/Saved Parameters/main_parameters.json.")

    # Validating the save file's content
    if cfg:
        cfg = list_or_dict(cfg)
    else:
        print("No saved config found. Defaulting...")
        cfg = default_params()
        time.sleep(1)

    # Downloading ideas.txt from GitHub for prompt generation
    if not os.path.exists("/content/ideas.txt"):
        ideas_response = requests.get("https://raw.githubusercontent.com/ZicoDiegoRR/stable_diffusion_xl_colab_ui/refs/heads/main/ideas.txt")
        if ideas_response.status_code == 200:
            with open("/content/ideas.txt", "w") as ideas_file:
                ideas_file.write(ideas_response.text)
            with open("/content/ideas.txt", "r") as ideas_file:
                ideas_line = ideas_file.readlines()
            gpt2_pipe = pipe('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')
        else:
            print("Failed to download ideas.txt from GitHub.")
            ideas_line = []
            gpt2_pipe = None
    else:
        with open("/content/ideas.txt", "r") as ideas_file:
            ideas_line = ideas_file.readlines()
        gpt2_pipe = pipe('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')

    return cfg, ideas_line, gpt2_pipe
