import requests
import os
import re

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
            download_header = {"Authorization:" f"Bearer {hf_token}"}
        else:
            download_header = ""
        download_url = url
    else:
        download_header = ""
        download_url = url

    # Download
    if download_url:
        if download_header:
            download_req = requests.get(download_url, headers=download_header, stream=True)
        else:
            download_req = requests.get(download_url, stream=True)

        filename_content_disposition = download_req.headers.get("Content-Disposition")
        if filename_content_disposition:
            filename_find = re.search(r"filename='(.+)'", filename_content_disposition)
            if filename_find:
                download_filename = filename_find.group(1)
            else:
                download_filename = os.path.basename(url) + ".safetensors"
        else:
            download_filename = os.path.basename(url) + ".safetensors"

        # Save
        with open(f"{download_folder}/{download_filename}", "wb") as f:
            for chunk in download_req.iter_content(chunk_size=8192):
                if chunk:
                    f.write(download_req.chunk)

    # Return the path
    return f"{download_folder}/{download_filename}"
