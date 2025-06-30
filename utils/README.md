## Generation Logic Modules

This folder contains the core functional components that power the image generation process. Each script is responsible for a specific part of the workflow:

- `controlnet_loader.py` — Loads ControlNet's weight and preprocess the ControlNet's parameters.

- `downloader.py` — Handles downloading external resources required for generation.

- `embeddings_loader.py` — Loads embeddings or textual inversion models into the pipeline.

- `generate_prompt.py` — Generates or extends textual prompts for image generation.
  
- `get_controlnet_image.py` — Processes the ControlNet's images.

- `hires_fix.py` — Applies Hires.Fix to your images.

- `image_saver.py` — Saves generated images to Google Drive (if integration is enabled).

- `ip_adapter_loader.py` — Loads IP-Adapter and handles its images.

- `lora_loader.py` — Applies LoRA (Low-Rank Adaptation) weights to the model.

- `main.py` — Controls the image generation logic and flow.

- `modified_inference_realesrgan.py` — Performs image upscaling using Real-ESRGAN.

- `pipeline_selector.py` — Selects the appropriate generation pipeline based on UI input.
  
- `preprocess.py` — Loads the save file and the prompt generator and processes them.

- `run_generation.py` — Executes the core image generation process.

- `save_file_converter.py` — Converts old saves to the new format.

- `scheduler_selector.py` — Picks the appropriate schedulers based on inputs.

- `vae_loader.py` — Loads a Variational Autoencoder (VAE) into the inference pipeline.

---

This README was made by ChatGPT.
