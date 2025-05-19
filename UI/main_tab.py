from StableDiffusionXLColabUI.UI import all_widgets
import ipywidgets as widgets

class UIWrapper:
    def merge_final_phase(self, init, destination, index, text2img, img2img, controlnet):
        all_widgets.merge(init, destination, text2img, img2img, controlnet)
        
    def merge_first_phase(self, index, text2img, img2img, controlnet):
        self.merge_options.children = [self.send_text2img, self.send_img2img, self.send_controlnet]
    
    def __init__(self, cfg, ideas_line):
        # Instantiate other classes
        self.text2img = Text2ImgSettings()
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

        # Creating the ui
        self.ui = widgets.Tab()
        self.ui.children = [
            widgets.VBox([text2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([img2img.wrap_settings(), self.additional_widgets]),
            widgets.VBox([controlnet.wrap_settings, self.additional_widgets]),
            inpaint.wrap_settings(),
            lora.wrap_settings(),
            embeddings.wrap_settings(),
            ip.wrap_settings(),
            upscale.ersgan_settings,
            history.wrap_settings()
        ]
