import subprocess
import os

# Initialize Real-ESRGAN
def initialize_realesrgan():
    os.chdir("/content/RealESRGAN")
    subprocess.run(["python", "setup.py", "develop"])
    os.chdir("/content")

# Doing everything
def start():
    initialize_realesrgan()

    # Import the preprocess module and UI
    from StableDiffusionXLColabUI.utils import preprocess
    from StableDiffusionXLColabUI.UI.ui_wrapper import UIWrapper

    # Setting the environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Preprocess the save file, ideas.txt, and GPT-2
    cfg, ideas_line, gpt2_pipe, base_path = preprocess.run()

    # Initialize the UI
    colab_ui = UIWrapper(cfg, ideas_line, gpt2_pipe, base_path)
