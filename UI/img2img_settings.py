from StableDiffusionXLColabUI.utils import generate_prompt
import ipywidgets as widgets
import os

class Img2ImgSettings:
    # Collect every widget into a single VBox
    def wrap_settings(self):
        return widgets.VBox([
            self.prompts_section,
            self.image_resolution_section,
            self.generation_parameter_section,
            self.reference_image_section,
            self.scheduler_settings,
            self.vae_section,
        ])

    # Generate prompt
    def generate_prompt_on_click(self, ideas_line, gpt2_pipe):
        generated_prompt = generate_prompt.generate(self.prompt_widget.value, ideas_line, gpt2_pipe)
        self.prompt_widget.value = generated_prompt

    def return_widgets(self):
        return [
            self.prompt_widget,
            self.negative_prompt_widget,
            self.model_widget,
            self.width_slider,
            self.height_slider,
            self.steps_slider,
            self.scale_slider,
            self.clip_skip_slider,
            self.scheduler_dropdown,
            self.karras_bool,
            self.vpred_bool,
            self.sgmuniform_bool,
            self.res_betas_zero_snr,
            self.vae_link_widget,
            self.vae_config,
            self.reference_image_link_widget,
            self.denoising_strength_slider,
            self.batch_size,
        ]
          
    # Collect all values from the widgets and turn them into a single list
    def collect_values(self):
        return [
            self.prompt_widget.value,
            self.negative_prompt_widget.value,
            self.model_widget.value,
            self.width_slider.value,
            self.height_slider.value,
            self.steps_slider.value,
            self.scale_slider.value,
            self.clip_skip_slider.value,
            self.scheduler_dropdown.value,
            self.karras_bool.value,
            self.vpred_bool.value,
            self.sgmuniform_bool.value,
            self.res_betas_zero_snr.value,
            self.vae_link_widget.value,
            self.vae_config.value,
            self.reference_image_link_widget.value,
            self.denoising_strength_slider.value,
            self.batch_size.value,
        ]
        
     # Function to show or hide scheduler booleans
    def scheduler_dropdown_handler(self, change):
        if change["new"] != "Default (defaulting to the model)":
            self.scheduler_settings.children = [self.scheduler_dropdown, self.karras_bool, self.vpred_bool, self.sgmuniform_bool, self.res_betas_zero_snr, widgets.HTML(value="Rescaling the betas to have zero terminal SNR helps to achieve vibrant color, but not necessary.")]
        else:
            self.scheduler_settings.children = [self.scheduler_dropdown]

    # Function to handle image upload
    def reference_image_upload_handler(self, change):
        os.makedirs("/content/img2img/", exist_ok=True)
        for file_info in self.reference_image_upload_widget.value.items():
            ref_uploaded_image = file_info[1]["content"]
            with open("/content/img2img/temp.png", "wb") as up:
                up.write(ref_uploaded_image)
            self.reference_image_link_widget.value = "/content/img2img/temp.png"
            
    # Initialize widgets creation
    def __init__(self, cfg, ideas_line, gpt2_pipe):
        prompt_layout = widgets.Layout(width="50%")
        self.prompt_widget = widgets.Textarea(value=cfg[0] if cfg else "", placeholder="Enter the prompt here.", layout=prompt_layout)
        self.negative_prompt_widget = widgets.Textarea(value=cfg[1] if cfg else "", placeholder="What you don't want to see?", layout=prompt_layout)
        self.prompt_randomize_button = widgets.Button(description="🔄", layout=widgets.Layout(width="40px"))
        self.prompt_randomize_button_label = widgets.Label(value="Randomize or continue your prompt with GPT-2")
        self.prompt_randomize_button.on_click(lambda b: self.generate_prompt_on_click(ideas_line, gpt2_pipe))

        self.prompts_section = widgets.VBox()
        self.prompts_section.children = [
            widgets.HBox([
                widgets.Label(value="Prompt:", layout=prompt_layout),
                widgets.Label(value="Negative Prompt:", layout=prompt_layout)
            ]),
            widgets.HBox([
                self.prompt_widget, self.negative_prompt_widget
            ]),
            widgets.HBox([
                self.prompt_randomize_button, self.prompt_randomize_button_label
            ]),
        ]

        self.model_widget = widgets.Text(value=cfg[2] if cfg else "", placeholder="HF's repository or direct URL")

        self.width_slider = widgets.IntSlider(min=512, max=1536, step=64, value=cfg[3] if cfg else 1024, description="Width")
        self.height_slider = widgets.IntSlider(min=512, max=1536, step=64, value=cfg[4] if cfg else 1024, description="Height")
        self.image_resolution_section = widgets.HBox([self.width_slider, self.height_slider])

        self.batch_size = widgets.IntText(value=cfg[17] if cfg else 1, description="Batch size")
        self.steps_slider = widgets.IntText(value=cfg[5] if cfg else 12, description="Steps")
        self.scale_slider = widgets.FloatSlider(min=1, max=12, step=0.1, value=cfg[6] if cfg else 6, description="Scale")
        self.clip_skip_slider = widgets.IntSlider(min=0, max=12, step=1, value=cfg[7] if cfg else 2, description="Clip Skip")
        self.generation_parameter_section = widgets.VBox([widgets.HBox([self.steps_slider, self.batch_size]), widgets.HBox([self.scale_slider, self.clip_skip_slider])])

        self.scheduler_dropdown = widgets.Dropdown(
            options=[
                "Default (defaulting to the model)", "DPM++ 2M", "DPM++ 2M SDE",
                "DPM++ SDE", "DPM2", "DDPM",
                "DPM2 a", "DDIM", "PNDM", "Euler", "Euler a", "Heun", "LMS",
                "DEIS", "UniPC"
            ],
            value=cfg[8] if cfg else "Default (defaulting to the model)",
            description="Scheduler",
        )
        self.karras_bool = widgets.Checkbox(value=cfg[9] if cfg else False, description="Enable Karras")
        self.vpred_bool = widgets.Checkbox(value=cfg[10] if cfg else False, description="Enable V-prediction")
        self.sgmuniform_bool = widgets.Checkbox(value=cfg[11] if cfg else False, description="Enable SGMUniform")
        self.res_betas_zero_snr = widgets.Checkbox(value=cfg[12] if cfg else False, description="Rescale beta zero SNR")
        self.scheduler_settings = widgets.VBox([self.scheduler_dropdown])

        self.scheduler_dropdown.observe(self.scheduler_dropdown_handler, names="value")
        self.scheduler_dropdown_handler({"new": self.scheduler_dropdown.value})

        self.vae_link_widget = widgets.Text(value=cfg[13] if cfg else "", description="VAE", placeholder="VAE model link")
        self.vae_config = widgets.Text(value=cfg[14] if cfg else "", placeholder="VAE config link")
        self.vae_section = widgets.HBox([self.vae_link_widget, self.vae_config])

        self.reference_image_link_widget = widgets.Text(placeholder="Img2Img reference link", description="Reference Image", value=cfg[15] if cfg and not cfg[15].startswith("/content/img2img/") else "")
        self.reference_image_upload_widget = widgets.FileUpload(accept="image/*", multiple=False)
        self.denoising_strength_slider = widgets.FloatSlider(min=0.1, max=1, step=0.01, description="Denoising Strength", value=cfg[16] if cfg else 0.3)
        self.reference_image_section = widgets.VBox([widgets.HBox([self.reference_image_link_widget, self.reference_image_upload_widget]), widgets.HBox([self.denoising_strength_slider, widgets.HTML(value="Low value means similar to the original image.")])])
        self.reference_image_upload_widget.observe(self.reference_image_upload_handler, names="value")
