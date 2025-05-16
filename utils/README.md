This folder contains the functions of the main logic behind the generation process.

- `downloader.py`: To download stuff obviously
- `embeddings_loader.py`: To load embeddings or textual inversion
- `image_saver.py`: To save the generated image into your Google Drive if enabled
- `lora_loader.py`: To add the weight of LoRAs into the model
- `modified_inference_realesrgan.py`: To run the Real-ESRGAN image upscaling process.
- `pipeline_selector.py`: To pick the correct pipeline based on the selected pipeline from the UI
- `run_generation.py`: To start the generation process
- `vae_loader.py`: To load VAE model to the pipeline
