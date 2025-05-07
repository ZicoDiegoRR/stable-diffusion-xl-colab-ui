# Google Colab Non WebUI Stable Diffusion XL Image Generator

This is a simple Google Colab notebook made by a junior programmer for generating images using Stable Diffusion XL.

## Features
- **Image Generation:** Uses Transformers and Diffusers as the major components for image generation, along with ControlNet, Inpainting, and IP-Adapter.
- **Download and Load:** Ability to download and load images, checkpoint, LoRA weights, and VAE using direct URLs.
- **Google Drive Integration:** Can connect to your Google Drive or disable it.
- **User-Friendly UI:** Simplified complexity with an easy-to-understand user interface.
- **IPyWidgets:** Simplified UI so that you don't need to scroll up and down. (V2 and above only)
- **Built-in History System:** Can show all of your previously generated images in Google Drive without opening it, just with a few simple clicks. (V2.5 and above only)
- **CivitAI Token Support:** Pass your CivitAI token for additional functionality, but remember **never share it with anyone.**
- **Simplicity:** Consists of two cells only.
- **Written Guide:** Can keep you on track with the flow of this notebook.

## Usage
This notebook is designed for artistic purposes and to spark inspiration. **Please use it responsibly.** Creating deepfakes and uploading them online are strictly prohibited.

## Installation
Simply open the notebook on Google Colab to start using it. No installation is needed. Debugging and editing the code might be difficult due to the complex structure and my lack of knowledge in modular programming.
- **V1:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZicoDiegoRR/stable_diffusion_xl_colab_ui/blob/main/stable_diffusion_xl_v1.ipynb)
- **V2:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZicoDiegoRR/stable_diffusion_xl_colab_ui/blob/main/stable_diffusion_xl_v2.ipynb)
- **V2.5:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZicoDiegoRR/stable_diffusion_xl_colab_ui/blob/main/stable_diffusion_xl_v2.5.ipynb)

|                               | V1 | V2 | V2.5 |
|-------------------------------|----|----|------|
| Base logic                    | ✅  | ✅  | ✅    |
| IPyWidgets                    | ❌  | ✅  | ✅    |
| Saving and loading parameters | ❌  | ✅  | ✅    |
| Interactive UI                | ❌  | ✅  | ✅    |
| Linking widgets               | ❌  | ❌  | ✅    |
| History system                | ❌  | ❌  | ✅    |
| Upload images directly        | ❌  | ❌  | ✅    |

## Disclaimer
- This tool is intended for creating art and exploring creative image generation.
- **Use responsibly.** Please refrain from using it for any malicious or harmful activities.
- NSFW generation is supported, but do it at your own risk.
- It's **recommended** to watch or use the V1 notebook first before using the V2 and above.

## Preview
### V1:
The resolution is too big. [Consider checking it manually.](docs/v1/v1.png)
### V2:
![general_settings_v2](docs/v2/general_settings.png)
![advanced_settings_v2](docs/v2/advanced_settings.png)
### V2.5:
![general_settings_v2.5](docs/v2.5/general_settings.png)
![advanced_settings_v2.5](docs/v2.5/advanced_settings.png)
![history_v2.5](docs/v2.5/history.png)

## To-do List
- Adding textual inversion
- Implementing Inpainting using IPyCanvas
- Adding Hugging Face's token integration
- Implementing Img2Img pipeline
## License
This project is open-source and free to use under the MIT License.

---

This README was mostly made by ChatGPT. (I suck at creating markdown)
