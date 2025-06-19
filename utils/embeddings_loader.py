from StableDiffusionXLColabUI.utils import downloader
from safetensors.torch import load_file
import re
import os           

def search_for_match(element_in_list, keys):
    for element in element_in_list:
        if element in keys:
            return True
    return False

def tokens_to_unload(nested_list, keys):
    saved = nested_list
    unload_ti = []
    for element in nested_list:
        if not search_for_match(element, keys):
            unload_ti.append(element)
            saved.remove(element)

    return saved, unload_ti

def unload_embeddings(pipe, saved, tokens):
    if saved:
        saved_token, unload_ti = tokens_to_unload(saved, tokens)
    else:
        saved_token = []
        unload_ti = []
    
    if unload_ti:
        print("Unloading certain textual inversion weights...")
        unload_tokens = []
        for ti in unload_ti:
            unload_tokens += ti
        print(f"Unloading tokens...\n{unload_tokens}")
        try:
            pipe.unload_textual_inversion(tokens=ti, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            pipe.unload_textual_inversion(tokens=ti, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
        except Exception as e:
            print(f"Unable to unload {ti}. Reason: {e}")

    return saved_token

def load_textual_inversion_from_link(pipe, link, token, name, embeddings_tokens):
    filtered_tokens = unload_embeddings(pipe, embeddings_tokens, token)

    # Loading the weight into the tokenizers and the text encoders
    loaded_name = []
    for embed, tag, name in zip(link, token, name):
        try:
            if token not in list(pipe.tokenizer.get_added_vocab().keys()):
                 # Getting the previously-loaded tokens
                old_tokens = list(pipe.tokenizer.get_added_vocab().keys())

                # Loading
                print(f"Loading {name}...")
                ti_dict = load_file(embed)
                pipe.load_textual_inversion(ti_dict["clip_g"], token=tag, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
                pipe.load_textual_inversion(ti_dict["clip_l"], token=tag, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

                # Getting the newly-added tokens, even the duplicates
                new_tokens = [keyword for keyword in pipe.tokenizer.get_added_vocab().keys() if keyword not in old_tokens and keyword != "<|endoftext|>" and keyword != "<|startoftext|>"]
                filtered_tokens.append(new_tokens)
                loaded_name.append(name)
                
        except Exception as e:
            print(f"Skipped {name}. Reason: {e}")
            if tag in list(pipe.tokenizer.get_added_vocab().keys()):
                loaded_name.append(name)

    # Output
    if [keyword for keyword in pipe.tokenizer.get_added_vocab().keys() if keyword not in old_tokens and keyword != "<|endoftext|>" and keyword != "<|startoftext|>"]:
        print("Loaded Textual Inversion or Embeddings:")
        for name in loaded_name:
            print(name)

    return filtered_tokens
    
def download_textual_inversion(pipe, link, token, embeddings_tokens, widget, hf_token, civit_token, base_path):
    # Download and handle duplication
    ti_list = []
    ti_path = []
    tokens = []
    unique_ti_urls = []
    
    for i, url in enumerate(link):
        textual_inversion_path = ""
        if url not in unique_ti_urls:
            if url.startswith("https://") or url.startswith("http://"):
                textual_inversion_path = downloader.download_file(url, "Embeddings", hf_token, civit_token, base_path)
            else:
                if url.startswith("/content/LoRAs/"):
                    ti_check = os.path.basename(url)
                else:
                    ti_check = url
                textual_inversion_path = downloader.download_file(url, "Embeddings", hf_token, civit_token, base_path)

            if textual_inversion_path and token[i] and not token[i].isspace():
                unique_ti_urls.append(url)
                ti_path.append(textual_inversion_path)

                split_filename, _ = os.path.splitext(os.path.basename(textual_inversion_path))
                ti_list.append(split_filename)
                tokens.append(token[i])

                widget_value = widget.value.replace(url, split_filename)
                widget.value = widget_value
            else: 
                if not token[i] or token[i].isspace() and url:
                    print(f"The token for {url} is empty or contains whitespaces only. This isn't supported in textual inversion.")
                if not textual_inversion_path and url:
                    print(f"It seems like {url} is an invalid path or doesn't exist. Make sure to put a correct path to ensure the weight being loaded correctly.")
                print(f"Skipped {url}.")

    return load_textual_inversion_from_link(pipe, ti_path, tokens, ti_list, embeddings_tokens)
        
def process(pipe, link, token, embeddings_tokens, widget, hf_token, civit_token, base_path):
    # Preprocessing the urls and weight before downloading
    ti_links = re.split(r"\s*,\s*", link)
    ti_tokens = re.split(r"\s*,\s*", token)
    
    os.makedirs("/content/Embeddings", exist_ok=True)

    return download_textual_inversion(pipe, ti_links, ti_tokens, embeddings_tokens, widget, hf_token, civit_token, base_path)
