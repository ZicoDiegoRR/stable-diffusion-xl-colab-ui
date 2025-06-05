from tqdm import tqdm
import requests
import os
import re

def is_exist(folder, name, type):
    if name.endswith(".safetensors") and not name.startswith(("https://", "http://", "/content")):
        subfolder, _ = os.path.splitext(name)
        weight_file = name
    elif name.startswith(("https://", "http://")):
        return False
    else:
        parts = name.strip("/").split("/")
        if len(parts) >= 2:
            subfolder = parts[-2]
            weight_file = parts[-1]
        else:
            subfolder = "unknown"
            weight_file = name

    full_path = f"{folder}/{type}/{subfolder}" if type == "VAE" else f"{folder}/{type}/{weight_file}"
    
    if not os.path.exists(full_path):
        return False
    if type == "VAE":
        return bool(os.listdir(full_path))
    return True

def download_file(url, type, hf_token, civit_token):
    # Folder creation if not exist
    download_folder = f"/content/{type}"
    os.makedirs(download_folder, exist_ok=True)

    # Handling the url based on the given server
    if "civitai.com" in url:
        download_header = ""
        if civit_token:
            if "?" in url or "&" in url:
                download_url = f"{url}&token={civit_token}"
            else:
                download_url = url + f"token={civit_token}"
        else:
            download_url = url
    elif "huggingface.co" in url:
        if hf_token:
            download_header = {"Authorization": f"Bearer {hf_token}"}
        else:
            download_header = ""
        download_url = url
    else:
        download_header = ""
        download_url = url

    # Download
    if download_url and not is_exist(download_folder, url, type):
        if download_header:
            download_req = requests.get(download_url, headers=download_header, stream=True)
        else:
            download_req = requests.get(download_url, stream=True)

        filename_content_disposition = download_req.headers.get("Content-Disposition")
        file_total_size = int(download_req.headers.get("content-length", 0))
        if filename_content_disposition:
            filename_find = re.search(r"filename=['\"]?([^'\"]+)['\"]?", filename_content_disposition)
            if filename_find:
                download_filename = filename_find.group(1)
            else:
                download_filename = os.path.basename(url) + ".safetensors"
        else:
            download_filename = os.path.basename(url) + ".safetensors"

        full_path = f"{download_folder}/{download_filename}"
        
        # Save
        with open(full_path, "wb") as f, tqdm(
            desc=download_filename,
            total=file_total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in download_req.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        full_path = url

    # Return the path
    return full_path
