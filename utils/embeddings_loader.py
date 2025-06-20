from StableDiffusionXLColabUI.utils import downloader
from safetensors.torch import load_file
import re
import os           

def get_vocab(pipe):
    return pipe.tokenizer.get_added_vocab().keys()

def search_for_match(element_in_list, keys):
    for element in element_in_list:
        if element in keys:
            return True
    return False

def unload_embeddings(pipe, saved, tokens):
    saved_filtered = [element for element in saved if search_for_match(element, tokens)]
    unload_ti = [element for element in saved if not search_for_match(element, tokens)]
    
    if unload_ti:
        print("Unloading certain textual inversion weights...")
        unload_tokens = []
        for ti in unload_ti:
            unload_tokens += ti
        print(f"Unloading tokens...\n{unload_tokens}")
        try:
            pipe.unload_textual_inversion(tokens=unload_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            pipe.unload_textual_inversion(tokens=unload_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
        except Exception as e:
            failed_tokens_to_unload = list(set(unload_tokens) - set(get_vocab(pipe)) - {"<|startoftext|>", "<|endoftext|>"})
            print(f"Unable to unload: \n{failed_tokens_to_unload} \nReason: {e}")

    return saved_filtered

def load_textual_inversion_from_link(pipe, link, token, name, embeddings_tokens):
    filtered_tokens = unload_embeddings(pipe, embeddings_tokens, token)
    
    # Loading the weight into the tokenizers and the text encoders
    loaded_name = []
    for path, activation_token, weight_name in zip(link, token, name):
        try:
            if activation_token not in list(get_vocab(pipe)):
                 # Getting the previously-loaded tokens
                old_tokens = list(get_vocab(pipe))

                # Loading
                print(f"Loading {weight_name}...")
                ti_dict = load_file(path)
                pipe.load_textual_inversion(ti_dict["clip_g"], token=activation_token, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
                pipe.load_textual_inversion(ti_dict["clip_l"], token=activation_token, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

                # Getting the newly-added tokens, even the duplicates
                new_tokens = list(set(get_vocab(pipe)) - set(old_tokens) - {"<|startoftext|>", "<|endoftext|>"})
                filtered_tokens.append(new_tokens)
                loaded_name.append(weight_name)
                
        except Exception as e:
            print(f"Skipped {weight_name}. Reason: {e}")
            if activation_token in list(get_vocab(pipe)):
                loaded_name.append(weight_name)

    # Output
    if list(set(get_vocab(pipe)) - {"<|startoftext|>", "<|endoftext|>"}):
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
