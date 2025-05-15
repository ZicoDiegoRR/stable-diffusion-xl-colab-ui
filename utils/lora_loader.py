from StableDiffusionXLColabUI.utils import downloader
import re
import os

def load_downloaded_lora(pipe, link, scales, names):
    scale_list = []
    name_list = []
    for file_path, scale, name in zip(link, scales, names):
        try:
            pipe.load_lora_weights(path, adapter_name=name)
            scale_list.append(scale)
            name_list.append(name)
        except Exception as e:
            print(f"Skipped {name}. Reason: {e}")
    pipe.set_adapters(name_list, adapter_weights=scale_list)

def download_lora(pipe, link, scale, hf_token, civit_token):
    lora_names = []
    lora_path = []
    scales = []
    unique_lora_urls = []

    for i, url in enumerate(link):
        if url not in unique_lora_urls:
            if url.startswith("https://") or url.startswith("http://"):
                lora_file_path = downloader.download_file(url, "LoRAs", hf_token, civit_token)
            else:
                lora_file_path = f"/content/LoRAs/{url}"
            unique_lora_urls.append(url)
            lora_path.append(lora_file_path)

            split_lora_name, _ = os.splitext(os.path.basename(lora_file_path))
            lora_names.append(split_lora_name)
            scales.append(scale[i])

    load_downloaded_lora(pipe, lora_path, scales, lora_names)

def process_url(pipe, link, scale, hf_token, civit_token):
    os.makedirs("/content/LoRAs", exist_ok=True)
    lora_links = [word for word in re.split(r"\s*,\s*", link)]
    lora_scales = [float(num) for num in re.split(r"\s*,\s*", scale)]
    
    download_lora(pipe, lora_links, lora_scales, hf_token, civit_token)
