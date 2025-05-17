'''
from StableDiffusionXLColabUI.UI.text2img_settings import Text2ImgSettings
from StableDiffusionXLColabUI.UI.img2img_settings import Img2ImgSettings
from StableDiffusionXLColabUI.UI.controlnet_settings import ControlNetSettings
from StableDiffusionXLColabUI.UI.inpainting_settings import InpaintingSettings
from StableDiffusionXLColabUI.UI.ip_adapter_settings import IPAdapterLoader
from StableDiffusionXLColabUI.UI.lora_settings import LoRALoader
from StableDiffusionXLColabUI.UI.textual_inversion_settings import TextualInversionLoader
'''

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
  
