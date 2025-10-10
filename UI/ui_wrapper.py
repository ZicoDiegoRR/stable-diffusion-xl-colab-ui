from StableDiffusionXLColabUI.utils import modified_inference_realesrgan, main, downloader
from StableDiffusionXLColabUI.UI.textual_inversion_settings import TextualInversionLoader
from StableDiffusionXLColabUI.UI.reset_and_generate import ResetGenerateSettings
from StableDiffusionXLColabUI.UI.controlnet_settings import ControlNetSettings
from StableDiffusionXLColabUI.UI.inpainting_settings import InpaintingSettings
from StableDiffusionXLColabUI.UI.ip_adapter_settings import IPAdapterLoader
from StableDiffusionXLColabUI.UI.text2img_settings import Text2ImgSettings
from StableDiffusionXLColabUI.UI.img2img_settings import Img2ImgSettings
from StableDiffusionXLColabUI.UI.preset_system import PresetSystem
from StableDiffusionXLColabUI.UI.lora_settings import LoRALoader
from StableDiffusionXLColabUI.UI.history import HistorySystem
from StableDiffusionXLColabUI.UI import all_widgets
import ipywidgets as widgets
import json
import os

def load_param(filename):
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        return {}

class UIWrapper:
    # Unused, but could be used for later
    def get_tab_index(self):
        return self.ui_tab.selected_index

    def select_class(self, index):
        if index == 0:
            return self.text2img
        elif index == 1:
            return self.img2img
        elif index == 2:
            return self.controlnet
        elif index == 3:
            return self.inpaint

    def select_key(self, index):
        if index == 0:
            return "text2img" 
        if index == 1:
            return "img2img"
        if index == 2:
            return "controlnet"
        if index == 3:
            return "inpaint"

    # To restart the runtime when users click the "Restart" button
    def restart(self):
        os.kill(os.getpid(), 9)

    # Hide the generate and send buttons or show them
    def checking_the_selected_tab_index(self, change): 
        self.tab_selected_index = change["new"]
        bool_value = self.draw
            
        if self.tab_selected_index <= 3 or self.tab_selected_index == 7:
            self.merge_button.disabled = False
            self.reset_generate.submit_button_widget.disabled = bool_value
            if self.tab_selected_index == 7:
                self.merge_button.disabled = True
        else:
            self.merge_button.disabled = True
            self.reset_generate.submit_button_widget.disabled = True

    def refresh_vae_selection(self):
        self.text2img.vae_link_widget.options = self.text2img.refresh_model()
        self.img2img.vae_link_widget.options = self.img2img.refresh_model()
        self.controlnet.vae_link_widget.options = self.controlnet.refresh_model()
        self.inpaint.vae_link_widget.options = self.inpaint.refresh_model()

    # Reload the combobox options
    def refresh_model(self):
        saved_models = load_param(f"{self.base_path}/Saved Parameters/URL/urls.json").get("Checkpoint")
        saved_hf_models = saved_models["hugging_face"] if saved_models and "hugging_face" in saved_models else []
        if not saved_models:
            model_options = []
        else:
            model_options = list(saved_models["keyname_to_url"].keys())
        self.model_widget.options = model_options + saved_hf_models
        
    # Displaying the submit button and resetting the history
    def reload_submit_button(self):
        if not self.is_downloading:
            self.submit_settings.layout.visibility = "visible"
            self.history.history_update(
                self.text2img,
                self.img2img,
                self.controlnet,
                self.inpaint,
                self.ip,
                self.lora,
                self.embeddings,
                self.upscaler,
                self.ui_tab,
                self.base_path
            )

    # Running the image generation
    def generate_value(self, index, text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
        for widget in [text2img.model_widget, img2img.model_widget, controlnet.model_widget]:
            widget.value = self.model_widget.value
            
        values_dictionary_for_generation = all_widgets.import_values(text2img, img2img, controlnet, inpaint, ip, lora, embeddings)
        widgets_dictionary_for_generation = all_widgets.import_widgets(text2img, img2img, controlnet, inpaint, ip, lora, embeddings)
        with self.generation_output:
            if index <= 3:
                key = self.select_key(index)
                selected_class = self.select_class(index)
                self.value_list = values_dictionary_for_generation[key]
                if not self.has_load_model:
                    self.value_list[2] = self.model_widget.value
                    
                self.submit_settings.layout.visibility = "hidden"
                self.model_output.clear_output()
                
                if text2img.return_hires_settings()[0][0] and index == 0:
                    hires = True
                    hires_values = text2img.return_hires_settings()[1]
                    if text2img.return_hires_settings()[1][0] == "Real-ESRGAN":
                        hires_values[0] = self.upscaler
                else:
                    hires = None
                    hires_values = None

                try:
                    main.run(self.value_list, 
                             values_dictionary_for_generation["lora"], 
                             values_dictionary_for_generation["embeddings"], 
                             values_dictionary_for_generation["ip"],
                             self.hf_token.value, 
                             self.civit_token.value, 
                             self.generation_output, 
                             [self.seed, self.freeze.value],
                             values_dictionary_for_generation,
                             [
                                [selected_class.vae_link_widget, selected_class.vae_config],
                                selected_class.model_widget,
                                self.lora.lora_urls_widget,
                                self.embeddings.ti_urls_widget,
                             ],
                             self.base_path,
                             self.controlnet.return_get_image_class(),
                             self.main_parameter,
                             hires,
                             hires_values,
                    )
                    self.has_load_model = True
                except Exception as e:
                    print(f"Generation interrupted by an error.\n Error: {e}")
    
                # Unused failsafe, but could be useful in rare moments
                if self.model_widget.value.startswith(("https://", "http://")):
                    self.model_widget.value, _ = os.path.splitext(os.path.basename(downloader.download_file(self.model_widget.value, "Checkpoint", self.hf_token.value, self.civit_token.value, self.base_path)))
    
                self.refresh_model()
                self.reload_submit_button()
                self.refresh_vae_selection()
                self.lora.construct(self.lora.lora_urls_widget.value)
                self.embeddings.construct(self.embeddings.ti_urls_widget.value)
            elif index == 7:
                self.submit_settings.layout.visibility = "hidden"
                self.upscaler.execute_realesrgan(self.generation_output)
                self.reload_submit_button()

    def load_always(self, change):
        if change["new"]:
            self.main_parameter = change["new"]
            self.preset_system.load_preset_on_click(
                change["new"],
                self.text2img,
                self.img2img,
                self.controlnet,
                self.inpaint,
                self.ip,
                self.lora,
                self.embeddings,
            )

    def load_custom_main(self, change):
        if change["new"]:
            self.main_parameter = self.preset_system.load_preset_selection_dropdown.value
            self.load_always({"new": self.main_parameter})
            self.preset_system.load_preset_selection_dropdown.observe(self.load_always, names="value")
        else:
            self.main_parameter = "main_parameters"
            self.preset_system.load_preset_selection_dropdown.unobserve(self.load_always, names="value")

    # Download models from model widget
    def load_model(self, url, hf_token, civit_token, base_path):
        # Initialize
        self.model_output.clear_output()
        progress_label = widgets.Label(value="Downloading...")
        progress_bar = widgets.IntProgress(value=0, min=0, max=100)
        warning_label = widgets.Label(value="The UI might be unresponsive at the moment. Please wait...")
        self.model_settings.children = [
            progress_label,
            progress_bar,
            self.model_label,
            widgets.HBox([
                self.model_widget, self.model_load_widget
            ]),
            warning_label,
        ]

        # Download
        self.loaded_model, _ = os.path.splitext(os.path.basename(downloader.download_file(url, "Checkpoint", hf_token, civit_token, base_path, tqdm=False, widget=progress_bar)))
        self.refresh_model()
        self.model_widget.value = self.loaded_model 
        self.model_settings.children = [
            self.model_output,
            self.model_label,
            widgets.HBox([
                self.model_widget, self.model_load_widget
            ]),
        ] if not self.has_load_model else [
            self.model_output,
            self.model_label,
            widgets.HBox([
                self.model_widget, self.model_load_widget
            ]),
            self.restart_button,
        ]

        # Output
        if url.count("/") == 1: # For Hugging Face's models
            output_msg = f"{self.loaded_model} is a Hugging Face model's format."
        else: # For any non-Hugging Face's models
            output_msg = f"{self.loaded_model} has been downloaded."

        if not self.has_load_model:
            follow_up_msg = "Click `Generate` to use it."
        else:
            follow_up_msg = "Restart the runtime to apply changes of the model."
            
        with self.model_output:
            print(f"{output_msg} {follow_up_msg}")
    
    # Final phase of merging a pipeline's general parameters to the selected pipeline
    def merge_final_phase(self, init, destination, index, text2img, img2img, controlnet, inpaint): # Doing merging
        if destination != "back":
            all_widgets.merge(init, destination, text2img, img2img, controlnet, inpaint)
            if destination == "text2img":
                self.ui_tab.selected_index = 0
            elif destination == "img2img":
                self.ui_tab.selected_index = 1
            elif destination == "controlnet":
                self.ui_tab.selected_index = 2
            elif destination == "inpaint":
                self.ui_tab.selected_index = 3
        self.merge_options.children = [self.merge_button]

    # First phase of merging a pipeline's general parameters to the selected pipeline
    def merge_first_phase(self, index, text2img, img2img, controlnet, inpaint): # Giving options
        merge_buttons = [self.send_text2img, self.send_img2img, self.send_controlnet, self.send_inpaint]
        merge_buttons.pop(index)

        self.merge_buttons_options.children = []
        for button in merge_buttons:
            self.merge_buttons_options.children += (button,)
        
        self.merge_options.children = [widgets.HBox([self.merge_buttons_options]), self.merge_back]
        type_for_init = self.select_key(index)

        self.send_text2img._click_handlers.callbacks.clear()
        self.send_img2img._click_handlers.callbacks.clear()
        self.send_controlnet._click_handlers.callbacks.clear()
        self.merge_back._click_handlers.callbacks.clear()
        
        self.send_text2img.on_click(lambda b: self.merge_final_phase(type_for_init, "text2img", index, self.text2img, self.img2img, self.controlnet, self.inpaint))
        self.send_img2img.on_click(lambda b: self.merge_final_phase(type_for_init, "img2img", index, self.text2img, self.img2img, self.controlnet, self.inpaint))
        self.send_controlnet.on_click(lambda b: self.merge_final_phase(type_for_init, "controlnet", index, self.text2img, self.img2img, self.controlnet, self.inpaint))
        self.send_inpaint.on_click(lambda b: self.merge_final_phase(type_for_init, "inpaint", index, self.text2img, self.img2img, self.controlnet, self.inpaint))
        self.merge_back.on_click(lambda b: self.merge_final_phase(type_for_init, "back", index, None, None, None))
    
    def __init__(self, cfg, ideas_line, gpt2_pipe, base_path): # cfg as a dictionary
        # Creating the tab
        self.ui_tab = widgets.Tab()
        self.generation_output = widgets.Output()
        self.value_list = []
        self.base_path = base_path
        self.draw = False
        self.main_parameter = "main_parameters"
        
        # Instantiate other classes
        self.text2img = Text2ImgSettings(cfg["text2img"], ideas_line, gpt2_pipe, base_path)
        self.img2img = Img2ImgSettings(cfg["img2img"], ideas_line, gpt2_pipe, base_path)
        self.controlnet = ControlNetSettings(cfg["controlnet"], ideas_line, gpt2_pipe, base_path)
        self.inpaint = InpaintingSettings(cfg["inpaint"], ideas_line, gpt2_pipe, base_path)
        self.ip = IPAdapterLoader(cfg["ip"])
        self.lora = LoRALoader(cfg["lora"], base_path)
        self.embeddings = TextualInversionLoader(cfg["embeddings"], base_path)
        self.upscaler = modified_inference_realesrgan.ESRGANWidget(base_path)
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
            base_path
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
            base_path,
        )
        
        # Wrapping widgets for seed
        self.seed = widgets.IntText(description="Seed")
        self.freeze = widgets.Checkbox(description="Use the same seed")
        self.seed_info = widgets.Label(value="You can input -1 and uncheck the 'Use the same seed' checkbox to use random seed.")

        self.seed_section = widgets.VBox([
            widgets.HBox([self.seed, self.freeze]),
            self.seed_info
        ])
        
        # Wrapping widgets for token and seed
        token_dict = load_param(f"{base_path}/Saved Parameters/Token/token.json")
        self.civit_label = widgets.Label(value="CivitAI:")
        self.hf_label = widgets.Label(value="Hugging Face:")

        self.civit_token = widgets.Password(placeholder="Avoid unauthorized error", value=token_dict.get("civit_token", ""))
        self.hf_token = widgets.Password(placeholder="Avoid unauthorized error", value=token_dict.get("hf_token", ""))

        self.token_section = widgets.HBox([
            widgets.VBox([self.civit_label, self.civit_token]),
            widgets.VBox([self.hf_label, self.hf_token])
        ])
        self.seed_and_token_section = widgets.VBox([self.seed_section, self.token_section])

        # Wrapping the model widget
        self.loaded_model = ""
        self.has_load_model = False
        self.is_downloading = False
        self.restart_button = widgets.Button(description="Restart")
        self.model_label = widgets.Label(value="Model:")
        self.model_widget = widgets.Combobox(
            value=self.text2img.model_widget.value,
            ensure_option=False
        )
        self.model_load_widget = widgets.Button(
            description="📥",
            layout=widgets.Layout(width="40px")
        )
        self.model_output = widgets.Output()

        self.refresh_model()
        self.model_load_widget.on_click(lambda b: self.load_model(self.model_widget.value, self.hf_token.value, self.civit_token.value, self.base_path))
        self.restart_button.on_click(lambda b: self.restart())

        self.model_settings = widgets.VBox([
            self.model_output,
            self.model_label,
            widgets.HBox([
                self.model_widget, self.model_load_widget
            ]),
        ])
        
       # Wrapping widgets for merge
        self.merge_button = widgets.Button(description="Send parameters")
        self.send_text2img = widgets.Button(description="Text-to-Image")
        self.send_img2img = widgets.Button(description="Image-to-Image")
        self.send_controlnet = widgets.Button(description="ControlNet")
        self.send_inpaint = widgets.Button(description="Inpainting")
        self.merge_back = widgets.Button(description="Back", layout=widgets.Layout(width="100%"))

        self.merge_buttons_options = widgets.HBox()
        self.merge_options = widgets.VBox([self.merge_button])
        self.merge_options.layout = widgets.Layout(margin='0 0 0 auto')

        # Wrapping widgets for reset and merge
        self.reset_settings = self.reset_generate.wrap_settings("reset")
        self.reset_and_send_section = widgets.HBox([self.reset_settings, self.merge_options])

        # Wrapping additional widgets
        self.additional_widgets = widgets.VBox([self.seed_and_token_section, self.reset_and_send_section])

        # Wrapping the Inpainting
        self.inpaint_output = widgets.Output()

        # Creating the UI
        border = widgets.HTML(value="<hr>")
        self.ui_tab.children = [
            widgets.VBox([self.model_settings, border, self.text2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.model_settings, border, self.img2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.model_settings, self.controlnet.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.inpaint_output, self.model_settings, border, self.inpaint.wrap_settings(), self.additional_widgets]),
            widgets.VBox([self.lora.wrap_settings(), self.token_section]),
            widgets.VBox([self.embeddings.wrap_settings(), self.token_section]),
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

        self.ui = widgets.VBox([self.ui_tab, self.ui_bottom])

        self.ui_tab.observe(self.checking_the_selected_tab_index, names="selected_index")
        self.checking_the_selected_tab_index({"name": "selected_index", "new": self.ui_tab.selected_index, "old": None, "type": "change", "owner": self.ui_tab})

        self.merge_button.on_click(lambda b: self.merge_first_phase(
            self.ui_tab.selected_index, self.text2img, self.img2img, self.controlnet, self.inpaint
        ))
        self.preset_system.use_as_main.observe(self.load_custom_main, names="value")
        
