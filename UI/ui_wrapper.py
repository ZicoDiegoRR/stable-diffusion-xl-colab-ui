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
from StableDiffusionXLColabUI.utils import modified_inference_realesrgan, main
from StableDiffusionXLColabUI.UI import all_widgets
from IPython.display import display, clear_output
import ipywidgets as widgets

class UIWrapper:
    def reload_submit_button(self):
        self.submit_settings.layout.visibility = "visible"
        
    def generate_value(self, index, text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
        values_dictionry_for_generation = all_widgets.import_values(text2img, img2img, controlnet, inpaint, ip, lora, embeddings)
        if index == 3:
            print("Inpainting is currently unavailable in this version. Please refer to the 'Legacy' version of this notebook. Sorry for the inconvenience.")
        elif index < 3:
            key = "text2img" if index == 0 else "img2img" if index == 1 else "controlnet" if index == 2
            self.value_list = values_dictionry_for_generation[key]
            self.submit_settings.layout.visibility = "hidden"
            main.run(self.value_list, self.hf_token, self.civit_token, self.ui, [self.seed, self.freeze.value])
            self.reload_submit_button()
        elif index == 7:
            self.submit_settings.layout.visibility = "hidden"
            self.upscaler.execute_realesrgan()
            self.reload_submit_button()
    
    def get_tab_index(self):
        return self.ui_tab.selected_index
    
    def merge_final_phase(self, init, destination, index, text2img, img2img, controlnet): # Doing merging
        all_widgets.merge(init, destination, text2img, img2img, controlnet)
        if destination == "text2img":
            self.ui_tab.selected_index = 0
        elif destination == "img2img":
            self.ui_tab.selected_index = 1
        elif destination == "controlnet":
            self.ui_tab.selected_index = 2
        
    def merge_first_phase(self, index, text2img, img2img, controlnet): # Giving options
        self.merge_options.children = [self.send_text2img, self.send_img2img, self.send_controlnet]
        type_for_init = "text2img" if index == 0 else "img2img" if index == 1 else "controlnet" if index == 2

        self.send_text2img._click_handlers.callbacks.clear()
        self.send_img2img._click_handlers.callbacks.clear()
        self.send_controlnet._click_handlers.callbacks.clear()
        
        self.send_text2img.on_click(lambda b: self.merge_first_phase(type_for_init, "text2img", self.text2img, self.img2img, self.controlnet))
        self.send_img2img.on_click(lambda b: self.merge_first_phase(type_for_init, "img2img", self.text2img, self.img2img, self.controlnet))
        self.send_controlnet.on_click(lambda b: self.merge_first_phase(type_for_init, "controlnet", self.text2img, self.img2img, self.controlnet))

    def checking_the_selected_tab_index(self, change): # Hiding the generate and send button or showing them
        self.tab_selected_index = change["new"]
        if self.tab_selected_index > 3 and self.tab_selected_index!= 7:
            self.submit_settings.layout.visibility = "hidden"
            self.merge_options.layout.visibility = "hidden"
        elif self.tab_selected_index == 7:
            self.submit_settings.layout.visibility = "visible"
            self.merge_options.layout.visibility = "visible"
        else:
            self.submit_settings.layout.visibility = "visible"
            self.merge_options.layout.visibility = "visible"
    
    def __init__(self, cfg, ideas_line): # cfg as a dictionary
        # Creating the tab
        self.ui_tab = widgets.Tab()
        self.value_list = []
        
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
            self.ui_tab,
        )
        
        # Wrapping widgets for seed
        self.seed = widgets.Text(description="Seed")
        self.freeze = widgets.Checkbox(description="Use the same seed")
        self.seed_info = widgets.Label(value="You can input -1 seed or uncheck the 'Use the same seed' checkbox to use random seed.")

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
        self.reset_settings = self.reset_generate.wrap_settings("reset")
        self.reset_settings.layout = widgets.Layout(margin='0 0 0 auto')
        self.reset_and_send_section = widgets.HBox([self.merge_options, self.reset_settings])

        # Wrapping additional widgets
        self.additional_widgets = widgets.VBox([self.seed_and_token_section, self.reset_and_send_section])

        # Creating the UI
        self.ui_tab.children = [
            widgets.VBox([self.text2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.img2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.controlnet.wrap_settings(), self.additional_widgets]),
            self.inpaint.wrap_settings(),
            self.lora.wrap_settings(),
            self.embeddings.wrap_settings(),
            self.ip.wrap_settings(),
            self.upscaler.ersgan_settings,
            self.history.wrap_settings()
        ]
        ui_titles = ["Text-to-image ✍", "Image-to-image 🎨", "ControlNet 🖼️🔧", "Inpainting 🖼️🖌️", "LoRA Settings 📁🖌️", "Textual Inversion 📃🖌️", "IP-Adapter Settings 🖼️📝", "Image Upscaler 🖼️✨", "History 🔮📜"]
        for i, title in enumerate(ui_titles):
            self.ui_tab.set_title(i, title)

        
        self.submit_settings = self.reset_generate.wrap_settings("submit")
        self.preset_settings = self.preset_system.wrap_settings()
        self.ui_bottom = widgets.HBox([self.submit_settings, self.preset_settings])

        self.reset_generate.submit_button_widget.on_click(lambda b: self.generate_value(
            self.ui_tab.selected_index, 
            self.text2img,
            self.img2img,
            self.controlnet,
            self.inpaint,
            self.ip,
            self.lora,
            self.embeddings,
        ))

        self.ui = widgets.VBox([self.ui_tab, self.ui_bottom])

        self.ui_tab.observe(self.checking_the_selected_tab_index, names="selected_index")
        self.checking_the_selected_tab_index({"name": "selected_index", "new": self.ui_tab.selected_index, "old": None, "type": "change", "owner": self.ui_tab})

        self.merge_button.on_click(lambda b: self.merge_first_phase(self.ui_tab.selected_index, self.text2img, self.img2img, self.controlnet))
