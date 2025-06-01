# Google Colab Non WebUI Stable Diffusion XL Image Generator (Legacy Version)

This is a simple Google Colab notebook made by an 18-years-old junior programmer for generating images using Stable Diffusion XL.

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

**Release Versions:**

- **V1:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18cUHh7H1c4qUijSD50uQUNl0Fq8wq7nY)
- **V2:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CQeZADunh6tEZCBOleDkZY96osYkRk5I)
- **V2.5:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13sfGKPhbCvNon0rpVvImUwVpZ3lkcuit)

**Pre-release Versions:**

- **Alpha V1:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14Cwd4DUEj9y1uWrSnFIaX4qItdPmTiVq)
- **Beta V2:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d5X_kSjUIEA4vSy8a-QTFbVLk8Y7AhJQ)
- **Beta V3:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wPT8aMUyJsOGMPdMgUH7AsZyC-KD9RTN)

<details> <summary>Feature Table</summary>
  
| Features                                                                                   |  V1 |  V2  | V2.5 | Beta V3 |
|--------------------------------------------------------------------------------------------|-----|------|------|---------|
| Base pipelines (ControlNet, VAE, Inpainting, Text2Img)                                     | ✅  | ✅  | ✅   | ✅      |
| Base adapters (LoRA, IP-Adapter)                                                           | ✅  | ✅  | ✅   | ✅      |
| IPyWidgets                                                                                 | ❌  | ✅  | ✅   | ✅      |
| Saving and loading parameters                                                              | ❌  | ✅  | ✅   | ✅      |
| Interactive UI                                                                             | ❌  | ✅  | ✅   | ✅      |
| Linking widgets                                                                            | ❌  | ❌  | ✅   | ✅      |
| History system                                                                             | ❌  | ❌  | ✅   | ✅      |
| Upload images directly                                                                     | ❌  | ❌  | ✅   | ✅      |
| Image-to-image                                                                             | ❌  | ❌  | ❌   | ✅      |
| Textual inversion or embeddings                                                            | ❌  | ❌  | ❌   | ✅      |
| Send images from history to Image-to-image, ControlNet, Inpainting, and/or IP-Adapter      | ❌  | ❌  | ❌   | ✅      |
| Reset button (defaulting the parameters)                                                   | ❌  | ❌  | ❌   | ✅      |
| Compatibility with saved parameters from previous versions                                 | ❌  | ❌  | ❌   | ✅      |
| Preset system (saving and loading custom parameters)                                       | ❌  | ❌  | ❌   | ✅      |
| GPT-2 Prompt Generator                                                                     | ❌  | ❌  | ❌   | ✅      |
| Hugging Face token integration                                                             | ❌  | ❌  | ❌   | ✅      |
| Real-ESRGAN Image Upscaling                                                                | ❌  | ❌  | ❌   | ✅      |
</details>


## Disclaimer
- This tool is intended for creating art and exploring creative image generation.
- **Use responsibly.** Please refrain from using it for any malicious or harmful activities.
- NSFW generation is supported, but do it at your own risk.

## Preview

<details> <summary> <b>V1:</b> </summary> <br>
  
The resolution is too big. [Consider checking it manually.](docs/v1/v1.png) </details>

<details> <summary> <b>V2:</b> </summary> <br>
  
![general_settings_v2](docs/v2/general_settings.png)

![advanced_settings_v2](docs/v2/advanced_settings.png) </details>

<details> <summary> <b>V2.5:</b> </summary> <br>
  
![general_settings_v2.5](docs/v2.5/general_settings.png)

![advanced_settings_v2.5](docs/v2.5/advanced_settings.png)

![history_v2.5](docs/v2.5/history.png) </details>

## License
This project is open-source and free to use under the MIT License.

---

This README was mostly made by ChatGPT. (I suck at creating markdown)
