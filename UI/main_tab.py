from StableDiffusionXLColabUI.UI.text2img_settings import Text2ImgSettings
from StableDiffusionXLColabUI.UI.img2img_settings import Img2ImgSettings
from StableDiffusionXLColabUI.UI.controlnet_settings import ControlNetSettings
from StableDiffusionXLColabUI.UI.inpainting_settings import InpaintingSettings
from StableDiffusionXLColabUI.UI.ip_adapter_settings import IPAdapterLoader
from StableDiffusionXLColabUI.UI.lora_settings import LoRALoader
from StableDiffusionXLColabUI.UI.textual_inversion_settings import TextualInversionLoader
from StableDiffusionXLColabUI.UI.reset_and_generate import ResetGenerateSettings
from StableDiffusionXLColabUI.UI.preset_system import PresetSystem
from StableDiffusionXLColabUI.UI.history import HistorySystem
from StableDiffusionXLColabUI.utils import modified_inference_realesrgan
from StableDiffusionXLColabUI.UI import all_widgets
import ipywidgets as widgets

class UIWrapper:
    def merge_final_phase(self, init, destination, index, text2img, img2img, controlnet):
        all_widgets.merge(init, destination, text2img, img2img, controlnet)
        
    def merge_first_phase(self, index, text2img, img2img, controlnet):
        self.merge_options.children = [self.send_text2img, self.send_img2img, self.send_controlnet]
    
    def __init__(self, cfg, ideas_line): # cfg as a dictionary
        # Creating the tab
        self.ui = widgets.Tab()
        
        # Instantiate other classes
        self.text2img = Text2ImgSettings(cfg["text2img"], ideas_line)
        self.img2img = Img2ImgSettings(cfg["img2img"], ideas_line)
        self.controlnet = ControlNetSettings(cfg["controlnet"], ideas_line)
        self.inpaint = InpaintingSettings(cfg["inpaint"])
        self.ip = IPAdapterLoader(cfg["ip"])
        self.lora = LoRALoader(cfg["lora"])
        self.embeddings = TextualInversionLoader(cfg["embeddings"])
        self.upscaler = modified_inference_realesrgan.ESRGANWidget()
        self.reset_generate = ResetGenerateSettings(
            self.text2img,
            self.img2img,
            self.controlnet,
            self.inpaint,
            self.ip,
            self.lora,
            self.embeddings,
        )
        self.preset_system = PresetSystem(
            self.text2img,
            self.img2img,
            self.controlnet,
            self.inpaint,
            self.ip,
            self.lora,
            self.embeddings,
        )
        self.history = HistorySystem(
            self.text2img,
            self.img2img,
            self.controlnet,
            self.inpaint,
            self.ip,
            self.lora,
            self.embeddings,
            self.upscaler,
            self.ui,
        )
        
        # Wrapping widgets for seed
        self.seed = widgets.Text(description="Seed")
        self.freeze = widgets.Checkbox(description="Use the same seed")
        self.seed_info = widgets.Label(value="You can input -1 seed or check the 'Use the same seed' checkbox to use the same seed.")

        self.seed_section = widgets.VBox([
            widgets.HBox([self.seed, self.freeze]),
            self.seed_info
        ])
        
        # Wrapping widgets for token and seed
        self.civit_label = widgets.Label(value="CivitAI:")
        self.hf_label = widgets.Label(value="Hugging Face:")

        self.civit_token = widgets.Text(placeholder="Avoid unauthorized error")
        self.hf_token = widgets.Text(placeholder="Avoid unauthorized error")

        self.token_section = widgets.HBox([
            widgets.VBox([self.civit_label, self.civit_token]),
            widgets.VBox([self.hf_label, self.hf_token])
        ])
        self.seed_and_token_section = widgets.VBox([self.seed_section, self.token_section])

       # Wrapping widgets for merge
        self.merge_button = widgets.Button(description="Send parameters")
        self.send_text2img = widgets.Button(description="Text-to-Image")
        self.send_img2img = widgets.Button(description="Image-to-Image")
        self.send_controlnet = widgets.Button(description="ControlNet")

        self.merge_options = widgets.HBox([self.merge_button])

        # Wrapping widgets for reset and merge
        reset_settings = reset.wrap_settings("reset")
        reset_settings.layout = widgets.Layout(margin='0 0 0 auto')
        self.reset_and_send_section = widgets.HBox([self.merge_options, reset_settings])

        # Wrapping additional widgets
        self.additional_widgets = widgets.VBox([self.seed_and_token_section, self.reset_and_send_section])

        # Creating the children
        self.ui.children = [
            widgets.VBox([self.text2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.img2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.controlnet.wrap_settings, self.additional_widgets]),
            self.inpaint.wrap_settings(),
            self.lora.wrap_settings(),
            self.embeddings.wrap_settings(),
            self.ip.wrap_settings(),
            self.upscale.ersgan_settings,
            self.history.wrap_settings()
        ]
