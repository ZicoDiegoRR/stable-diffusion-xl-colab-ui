from StableDiffusionXLColabUI.utils import preprocess
from StableDiffusionXLColabUI.UI.ui_wrapper import UIWrapper
import subprocess
import os

def start():
    # Initialize Real-ESRGAN
    os.chdir("/content/RealESRGAN")
    subprocess.run(["python", "setup.py", "develop"])
    os.chdir("/content")
    
    cfg, ideas_line, gpt2_pipe = preprocess.run() # Preprocess the save file, ideas.txt, and GPT-2
    colab_ui = UIWrapper(cfg, ideas_line, gpt2_pipe)# Doing everything
