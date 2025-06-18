from StableDiffusionXLColabUI.utils import downloader
import re
import os

def unload_lora(loaded, lora_names):
    unload_lora = []
    for lora in loaded:
        if lora and lora not in lora_names:
            unload_lora.append(lora)

    if unload_lora:
        for lora in unload_lora:
            print(f"Unloading {lora}")
            pipe.delete_adapters(lora)

def load_downloaded_lora(pipe, link, scales, names):
    scale_list = []
    name_list = []
    loaded_lora = pipe.get_active_adapters()
    if len(loaded_lora) < len(names):
        for i in range(len(name) - len(loaded_lora)):
            loaded_lora.append(None)

    unload_lora(loaded_lora, names)
    for file_path, scale, name in zip(link, scales, names):
        try:
            print(f"Loading {name}...")
            pipe.load_lora_weights(file_path, adapter_name=name)
            scale_list.append(scale)
            name_list.append(name)

        except Exception as e:
            print(f"Skipped {name}. Reason: {e}")
    
    if name_list:
        pipe.set_adapters(name_list, adapter_weights=scale_list)
        print("LoRA(s):")
        for name in name_list:
            print(name)

def download_lora(pipe, link, scale, widget, hf_token, civit_token, base_path):
    lora_names = []
    lora_paths = []
    scales = []
    unique_lora_urls = []

    for i, url in enumerate(link):
        lora_file_path = ""
        if url not in unique_lora_urls:
            if url.startswith("https://") or url.startswith("http://"):
                lora_file_path = downloader.download_file(url, "LoRAs", hf_token, civit_token, base_path)
            else:
                if url.startswith("/content/LoRAs/"):
                    lora_check = os.path.basename(url)
                else:
                    lora_check = url
                lora_file_path = downloader.download_file(lora_check, "LoRAs", hf_token, civit_token, base_path)
                        
            if lora_file_path:
                unique_lora_urls.append(url)
                lora_paths.append(lora_file_path)

                split_lora_name, _ = os.path.splitext(os.path.basename(lora_file_path))
                lora_names.append(split_lora_name)
                scales.append(scale[i])

                widget_value = widget.value.replace(url, split_lora_name)
                widget.value = widget_value
            else:
                print(f"It seems like {url} is an invalid path or doesn't exist. Make sure to put a correct path to ensure the weight being loaded correctly.")
                print(f"Skipped {url}.")

    load_downloaded_lora(pipe, lora_paths, scales, lora_names)

def process(pipe, link, scale, widget, hf_token, civit_token, base_path):
    os.makedirs("/content/LoRAs", exist_ok=True)
    lora_links = re.split(r"\s*,\s*", link)
    lora_scales = [float(num) for num in re.split(r"\s*,\s*", scale)]
    
    download_lora(pipe, lora_links, lora_scales, widget, hf_token, civit_token, base_path)
