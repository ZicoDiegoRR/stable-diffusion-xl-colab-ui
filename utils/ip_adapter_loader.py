from diffusers.utils import load_image, make_image_grid
import re

def load(
    pipeline,
    IP_Adapter,
    IP_Image_Link,
    IP_Adapter_Strength,
):
    # Loading the images
    adapter_image = []
    simple_Url = [word for word in re.split(r"\s*,\s*", IP_Image_Link) if word]
    for link in simple_Url:
        adapter_image.append(load_image(link))

    # Creating the display
    adapter_display = [element for element in adapter_image]
    if len(adapter_image) % 3 == 0:
        row = int(len(adapter_image)/3)
    else:
        row = int(len(adapter_image)/3) + 1
        for i in range(3*row - len(adapter_image)):
            adapter_display.append(load_image("https://huggingface.co/IDK-ab0ut/BFIDIW9W29NFJSKAOAOXDOKERJ29W/resolve/main/placeholder.png"))
    print("Image(s) for IP-Adapter:")
    display(make_image_grid([element.resize((1024, 1024)) for element in adapter_display], rows=row, cols=3))

    # Loading the images to the IP-Adapter
    image_embeds = [adapter_image]
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=IP_Adapter, low_cpu_mem_usage=True)
    pipeline.set_ip_adapter_scale(IP_Adapter_Strength)
    return image_embeds
