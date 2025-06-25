# UI Folder – Stable Diffusion XL Colab UI User Interface

This folder contains all the UI components and logic used in this **Stable Diffusion XL Colab UI** repository. Each file is modular, handling a specific part of the interface or parameter system. Below is a breakdown of what each file does.

---

## File Overview

| File Name                    | Description |
|-----------------------------|-------------|
| **`all_widgets.py`**        | Centralized access to all widgets across pipelines. Provides functions to collect current values, import them back, and handle parameter merging from one pipeline to another. |
| **`controlnet_settings.py`**| Handles the UI and logic for the ControlNet pipeline, including enabling/disabling conditions and managing ControlNet-specific inputs. |
| **`history.py`**            | Displays a history panel showing all previously generated images. Allows users to visually browse their output history without needing to open Google Drive. |
| **`img2img_settings.py`**   | Handles the UI and logic for the Image-to-Image pipeline, including input image settings, strength values, and generation controls. |
| **`inpainting_settings.py`**| Manages the Inpainting UI and logic.                                                                                            |
| **`ip_adapter_settings.py`**| Provides UI components and backend logic for using IP-Adapter in generation pipelines. |
| **`lora_settings.py`**      | Manages the LoRA (Low-Rank Adaptation) integration, allowing users to load and configure LoRA models via the UI. |
| **`mask_canvas.py`**| Handles the Inpainting canvas for creating mask images.                                                                                 |
| **`preset_system.py`**      | Provides a full preset management system: users can **save, load, rename, and delete** parameter presets through a structured UI. |
| **`reset_and_generate.py`**| Handles the **generate** and **reset** buttons. Includes logic to apply all parameters, submit them to the pipeline, or clear them. |
| **`text2img_settings.py`**  | Handles the UI and logic for the Text-to-Image pipeline, including prompt inputs, model settings, and generation configurations. |
| **`textual_inversion_settings.py`** | Similar to `lora_settings.py`, but focused on **textual inversion embeddings**—another form of model fine-tuning. |
| **`ui_wrapper.py`**         | The main UI orchestrator. Instantiates all components, links them together in a tabbed layout, and manages state transitions and interactions between modules. |

---

## Notes

- All modules are integrated under `ui_wrapper.py`, which serves as the root UI handler.
- All widgets are designed with Colab/IPyWidgets/IPyCanvas compatibility in mind.
- This README was made by ChatGPT.
