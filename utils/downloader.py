from tqdm import tqdm
import requests
import json
import os
import re

# Save urls.json
def save_param(path, data):
    with open(path, 'w') as file:
        json.dump(data, file)

# Load urls.json
def load_param(filename):
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        return {
            "VAE": {
                "keyname_to_url": {
                    "weight": {

                    },
                    "config": {
                    
                    }
                },
                "url_to_keyname": {
                    "weight": {
                    
                    },
                    "config": {
                    
                    }
                }
            },
            "Checkpoint": {
                "keyname_to_url": {
                    
                },
                "url_to_keyname": {
                    
                }
            },
            "LoRAs": {
                "keyname_to_url": {
                    
                },
                "url_to_keyname": {
                    
                }
            },
            "Embeddings": {
                "keyname_to_url": {
                    
                },
                "url_to_keyname": {
                    
                }
            },
        }

# Search for a match
def search(type, name):
    for dir in os.listdir(f"/content/{type}"):
        if name in dir:
            return dir
    return name

# Check if a path exists
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
            subfolder = name
            weight_file = search(type, name)

    full_path = f"{folder}/{subfolder}" if type == "VAE" else f"{folder}/{weight_file}"

    if type == "VAE":
        try:
          return bool(os.listdir(full_path))
        except FileNotFoundError:
          return False
    if not os.path.exists(full_path):
        return False
    return True

def download(url, type, hf_token, civit_token, key=None):
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
        
        if os.path.exists(full_path):
            return full_path
            
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

# Validate if the url has been downloaded before (even in previous instance)
def download_file(url, type, hf_token, civit_token, base_path, subfolder=None):
    # Load the dictionary from urls.json
    saved_urls = load_param(f"{base_path}/Saved Parameters/URL/urls.json")
    dict_type = saved_urls[type]

    # Select the key when loading VAE
    if subfolder:
        vae_key = "config"
    else:
        vae_key = "weight"
   
    # Handle URL input
    if (url.startswith("https://") or url.startswith("http://")) and not url.startswith("/content/gdrive/MyDrive"):
        key = dict_type.get("url_to_keyname").get(url) if type != "VAE" else dict_type.get("url_to_keyname").get(vae_key).get(url)
        if key:
            if is_exist(f"/content", key, type):
                returned_path = f"/content/{type}/{search(type, key)}"
            else:
                returned_path = download(url, type, hf_token, civit_token)
        else:
            returned_path = download(url, type, hf_token, civit_token)
            if type == "VAE":
                if vae_key == "weight":
                    vae_name, _ = os.path.splitext(os.path.basename(returned_path))
                elif vae_key == "config":
                    vae_name = subfolder

                saved_urls[type]["url_to_keyname"][vae_key][url] = vae_name
                saved_urls[type]["keyname_to_url"][vae_key][vae_name] = url
            else:
                file_name, _ = os.path.splitext(os.path.basename(returned_path))
                saved_urls[type]["url_to_keyname"][url] = file_name
                saved_urls[type]["keyname_to_url"][file_name] = url

    # Unused, but can handle file from Google Drive
    elif url.startswith("/content/gdrive/MyDrive"):
        returned_path = url

    # Handle key input
    else:
        key = url
        link = dict_type.get("keyname_to_url").get(key) if type != "VAE" else dict_type.get("keyname_to_url").get(vae_key).get(key)
        if link and not subfolder:
            if is_exist(f"/content", key, type):
                returned_path = f"/content/{type}/{search(type, key)}"
            else:
                returned_path = download(link, type, hf_token, civit_token)
        elif subfolder:
            if is_exist(f"/content/VAE/{subfolder}", "config", type):
                returned_path = f"/content/VAE/{subfolder}/config.json"
            else:
                returned_path = download(link, type, hf_token, civit_token)
        else:
            print(f"It seems like {url} doesn't exist in both /content/{type} directory and urls.json file. Is it a correct path?")
            returned_path = url

    save_param(f"{base_path}/Saved Parameters/URL/urls.json", saved_urls)

    return returned_path
