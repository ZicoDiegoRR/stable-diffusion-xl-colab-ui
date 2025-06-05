import subprocess
import os

# Import the preprocess module and UI
def import_ui():
    from StableDiffusionXLColabUI.utils import preprocess
    from StableDiffusionXLColabUI.UI.ui_wrapper import UIWrapper

# Initialize Real-ESRGAN
def initialize_realesrgan():
    os.chdir("/content/RealESRGAN")
    subprocess.run(["python", "setup.py", "develop"])
    os.chdir("/content")

    import_ui()

# Doing everything
def start():
    initialize_realesrgan()
    
    cfg, ideas_line, gpt2_pipe = preprocess.run() # Preprocess the save file, ideas.txt, and GPT-2
    colab_ui = UIWrapper(cfg, ideas_line, gpt2_pipe) # Initialize the UI
