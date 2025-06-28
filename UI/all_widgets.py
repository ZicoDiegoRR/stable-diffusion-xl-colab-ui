def import_widgets(text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
    widgets_dictionary = {
        "text2img": text2img.return_widgets(),
        "img2img": img2img.return_widgets(),
        "controlnet": controlnet.return_widgets(),
        "inpaint": inpaint.return_widgets(),
        "ip": ip.return_widgets(),
        "lora": lora.return_widgets(),
        "embeddings": embeddings.return_widgets(),
    }
    return widgets_dictionary

def import_values(text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
    values_dictionary = {
        "text2img": text2img.collect_values(),
        "img2img": img2img.collect_values(),
        "controlnet": controlnet.collect_values(),
        "inpaint": inpaint.collect_values(),
        "ip": ip.collect_values(),
        "lora": lora.collect_values(),
        "embeddings": embeddings.collect_values(),
    }
    return values_dictionary

def merge(init, destination, text2img, img2img, controlnet, inpaint):
    widgets_dictionary_for_merging = {
        "text2img": text2img.return_widgets(),
        "img2img": img2img.return_widgets(),
        "controlnet": controlnet.return_widgets(),
        "inpaint": inpaint.return_widgets(),
    }
    values_dictionary_for_merging = {
        "text2img": text2img.collect_values(),
        "img2img": img2img.collect_values(),
        "controlnet": controlnet.collect_values(),
        "inpaint": inpaint.collect_values(),
    }
    init_values = values_dictionary_for_merging[init]
    destination_widgets = widgets_dictionary_for_merging[destination]
    for i in range(15):
        destination_widgets[i].value = init_values[i]

    # For batch size
    destination_widgets[-1].value = init_values[-1]
