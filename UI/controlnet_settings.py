from StableDiffusionXLColabUI.utils import generate_prompt
import ipywidgets as widgets
import os

class ControlNetSettings:
    # Collect every widget into a single VBox
    def wrap_settings(self):
        return widgets.VBox([
            self.prompts_section,
            self.model_input_section,
            self.image_resolution_section,
            self.generation_parameter_section,
            self.scheduler_settings,
            self.vae_section,
            self.freeze_widget,
            self.token_section,
            widgets.HTML(value="For safety reason, your tokens <b>won't be saved</b>.")
        ])

    # Cheeck if the path is link
    def check_if_link(self, value):
        return value.startswith("https://") or value.startswith("http://")
        
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
            self.vae_link_widget.value if check_if_link(self.vae_link_widget.value) else "",
            self.vae_config.value if check_if_link(self.vae_config.value) else "",
            self.freeze_widget.value
        ]
        
     # Function to show or hide scheduler booleans
    def scheduler_dropdown_handler(self, change):
        if change["new"] != "Default (defaulting to the model)":
            self.scheduler_settings.children = [self.scheduler_dropdown, self.karras_bool, self.vpred_bool, self.sgmuniform_bool, self.res_betas_zero_snr, widgets.HTML(value="Rescaling the betas to have zero terminal SNR helps to achieve vibrant color, but not necessary.")]
        else:
            self.scheduler_settings.children = [self.scheduler_dropdown]

    def controlnet_preset_ref(self, value):
        if value == "Last Generated Text2Img":
            return ""
        elif value == "Last Generated ControlNet":
            return "controlnet"
        else:
            return "inpaint"

    def controlnet_dropdown_handler(self, type, value): # Function to change the image reference based on the selected option in the dropdown
        self.controlnet_url_widgets_list = [canny_link_widget, depth_map_link_widget, openpose_link_widget]
        self.controlnet_upload_widgets_list = [canny_upload, depth_upload, openpose_upload]

        controlnet_type = 0 if type == "canny" else 1 if type == "depth" else 2
        controlnet_children = list(canny_settings.children) if controlnet_type == 0 else list(depth_settings.children) if controlnet_type == 1 else list(openpose_settings.children)
        if value == "Link":
            if controlnet_upload_widgets_list[controlnet_type] in controlnet_children:
                controlnet_children.pop(2)
                controlnet_children.insert(2, controlnet_url_widgets_list[controlnet_type])
        elif value == "Upload":
            if controlnet_url_widgets_list[controlnet_type] in controlnet_children:
                controlnet_children.pop(2)
                controlnet_children.insert(2, controlnet_upload_widgets_list[controlnet_type])
        else:
            if controlnet_url_widgets_list[controlnet_type] in controlnet_children:
                controlnet_children.pop(2)
            for i in range(len(controlnet_url_widgets_list)):
                if controlnet_url_widgets_list[i] in controlnet_children:
                    controlnet_children.remove(controlnet_url_widgets_list[i])
                if controlnet_upload_widgets_list[i] in controlnet_children:
                    controlnet_children.remove(controlnet_upload_widgets_list[i])
                    
                if controlnet_type == 0:
                    self.canny_link_widget.value = self.controlnet_preset_ref(value)
                elif controlnet_type == 1:
                    self.depth_map_link_widget.value = self.controlnet_preset_ref(value)
                else:
                    self.openpose_link_widget.value = self.controlnet_preset_ref(value)

        if controlnet_type == 0:
            self.canny_settings.children = tuple(controlnet_children)
        elif controlnet_type == 1:
            self.depth_settings.children = tuple(controlnet_children)
        else:
            self.openpose_settings.children = tuple(controlnet_children)

    # Canny
    def canny_popup(self, change): # Function to display canny settings if true
        if change["new"]:
            self.canny_settings.children = [self.canny_toggle, self.canny_dropdown, self.canny_min_slider, self.canny_max_slider, self.canny_strength_slider]
        else:
            self.canny_settings.children = [self.canny_toggle]

    def canny_dropdown_handler(self, change): # Function to attach the canny dropdown to the controlnet dropdown handler
        self.controlnet_dropdown_handler("canny", change["new"])

    def canny_upload_handler(change): # Function to load the path of the uploaded image to the image link
        os.makedirs("/content/canny", exist_ok=True)
        for file_info in self.canny_upload.value.items():
            canny_uploaded_image = file_info[1]["content"]
            with open("/content/canny/temp.png", "wb") as up:
                up.write(canny_uploaded_image)
            self.canny_link_widget.value = "/content/canny/temp.png"
    # ________________________________________________________________________________________________________________________________________________________
    
    # Depth Map
    def depthmap_popup(self, change): # Function to display depth map settings if true
        if change["new"]:
            self.depth_settings.children = [self.depth_map_toggle, self.depthmap_dropdown, self.depth_strength_slider]
        else:
            self.depth_settings.children = [self.depth_map_toggle]

    def depthmap_dropdown_handler(self, change): # Function to attach the canny dropdown to the controlnet dropdown handler
        self.controlnet_dropdown_handler("depth", change["new"])

    def depthmap_upload_handler(change): # Function to load the path of the uploaded image to the image link
        os.makedirs("/content/depthmap/", exist_ok=True)
        for file_info in self.depth_upload.value.items():
            depth_uploaded_image = file_info[1]["content"]
            with open("/content/depthmap/temp.png", "wb") as up:
                up.write(depth_uploaded_image)
            self.depth_map_link_widget.value = "/content/depthmap/temp.png"
    
    # ________________________________________________________________________________________________________________________________________________________

    # Open Pose
    def openpose_popup(self, change):  # Function to display openpose settings if true
        if change["new"]:
            self.openpose_settings.children = [self.openpose_toggle, self.openpose_dropdown, self.openpose_strength_slider]
          else:
            self.openpose_settings.children = [self.openpose_toggle]

def openpose_dropdown_handler(change): # Function to attach the canny dropdown to the controlnet dropdown handler
  controlnet_dropdown_handler("openpose", change["new"])

def openpose_upload_handler(change): # Function to load the path of the uploaded image to the image link
  print(openpose_upload.value)
  if not os.path.exists("/content/openpose/"):
    os.mkdir("/content/openpose/")
  for file_info in openpose_upload.value.items():
    openpose_uploaded_image = file_info[1]["content"]
    with open("/content/openpose/temp.png", "wb") as up:
      up.write(openpose_uploaded_image)
  openpose_link_widget.value = "/content/openpose/temp.png"
    # ________________________________________________________________________________________________________________________________________________________
    # Initialize widgets creation
    def __init__(self, cfg, ideas_line):
        self.prompt_widget = widgets.Textarea(value=cfg[0] if cfg else "", placeholder="Enter your prompt here")
        self.negative_prompt_widget = widgets.Textarea(value=cfg[1] if cfg else "", placeholder="What you don't want to see?")
        self.prompt_randomize_button = widgets.Button(description="🔄", layout=widgets.Layout(width="40px"))
        self.prompt_randomize_button_label = widgets.Label(value="Randomize or continue your prompt with GPT-2")

        self.prompt_widget.layout.width = "50%"
        self.negative_prompt_widget.layout.width = "50%"
        self.prompt_randomize_button.on_click(lambda b: generate_prompt.generate(self.prompt_widget.value))

        self.prompts_section = widgets.HBox()
        self.prompts_section.children = [widgets.VBox([widgets.Label(value="Prompt:"), self.prompt_widget, widgets.HBox([self.prompt_randomize_button, self.prompt_randomize_button_label])]), widgets.VBox([widgets.Label(value="Negative prompt:"), self.negative_prompt_widget])] if ideas_line else [widgets.VBox([widgets.Label(value="Prompt:"), self.prompt_widget]), widgets.VBox([widgets.Label(value="Negative prompt:"), self.negative_prompt_widget])]

        self.model_widget = widgets.Text(value=cfg[2] if cfg else "", placeholder="HF's repository or direct URL")
        self.model_input_section = widgets.HBox([widgets.Label(value="Model link"), self.model_widget])

        self.width_slider = widgets.IntSlider(min=512, max=1536, step=64, value=cfg[3] if cfg else 1024, description="Width")
        self.height_slider = widgets.IntSlider(min=512, max=1536, step=64, value=cfg[4] if cfg else 1024, description="Height")
        self.image_resolution_section = widgets.HBox([self.width_slider, self.height_slider])

        self.steps_slider = widgets.IntText(value=cfg[5] if cfg else 12, description="Steps")
        self.scale_slider = widgets.FloatSlider(min=1, max=12, step=0.1, value=cfg[6] if cfg else 6, description="Scale")
        self.clip_skip_slider = widgets.IntSlider(min=0, max=12, step=1, value=cfg[7] if cfg else 2, description="Clip Skip")
        self.generation_parameter_section = widgets.VBox([self.steps_slider, widgets.HBox([self.scale_slider, self.clip_skip_slider])])

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

        self.civit_token_label = widgets.Label(value="CivitAI token:")
        self.token_widget = widgets.Text(placeholder="Avoid 401 error from CivitAI")
        
        self.hf_token_label = widgets.Label(value="Hugging Face token:")
        self.hf_token_widget = widgets.Text(placeholder="Avoid 401 error from HF")
        self.token_section = widgets.HBox([widgets.VBox([self.civit_token_label, self.token_widget]), widgets.VBox([self.hf_token_label, self.hf_token_widget])])

        self.freeze_widget = widgets.Checkbox(description="Use the same seed", value=cfg[15] if cfg else False)

        self.controlnet_dropdown_choice = ["Link", "Upload", "Last Generated Text2Img", "Last Generated ControlNet", "Last Generated Inpainting"]
        

canny_upload = widgets.FileUpload(accept="image/*", multiple=False)
canny_link_widget = widgets.Text(value=cfg[15] if cfg and (not cfg[15].startswith("/content/canny/") or os.path.exists("/content/canny/")) else "", description="Canny Link", placeholder="Image link")

canny_dropdown = widgets.Dropdown(options=controlnet_dropdown_choice, value=controlnet_dropdown_choice[2] if canny_link_widget.value == "" else controlnet_dropdown_choice[3] if canny_link_widget.value == "controlnet" else controlnet_dropdown_choice[4] if canny_link_widget.value == "inpaint" else controlnet_dropdown_choice[0], disabled=False, description="Reference Image")
canny_min_slider = widgets.IntSlider(min=10, max=500, step=5, value=cfg[13] if cfg else 100, description="Min Threshold")
canny_max_slider = widgets.IntSlider(min=100, max=750, step=5, value=cfg[14] if cfg else 240, description="Max Threshold")
canny_toggle = widgets.Checkbox(value=cfg[16] if cfg else False, description="Enable Canny")
canny_strength_slider = widgets.FloatSlider(min=0.1, max=1, step=0.1, value=cfg[17] if cfg else 0.7, description="Canny Strength")
canny_settings = widgets.VBox([canny_toggle])

canny_popup({"new": canny_toggle.value})
canny_upload.observe(canny_upload_handler, names="value")

depth_upload = widgets.FileUpload(accept="image/*", multiple=False)
depth_map_link_widget = widgets.Text(value=cfg[18] if cfg and (not cfg[18].startswith("/content/depthmap/") or os.path.exists("/content/depthmap/")) else "", description="DepthMap Link", placeholder="Image link")

depthmap_dropdown = widgets.Dropdown(options=controlnet_dropdown_choice, value=controlnet_dropdown_choice[2] if depth_map_link_widget.value == "" else controlnet_dropdown_choice[3] if depth_map_link_widget.value == "controlnet" else controlnet_dropdown_choice[4] if depth_map_link_widget.value == "inpaint" else controlnet_dropdown_choice[0], disabled=False, description="Reference Image")
depth_map_toggle = widgets.Checkbox(value=cfg[19] if cfg else False, description="Enable Depth Map")
depth_strength_slider = widgets.FloatSlider(min=0.1, max=1, step=0.1, value=cfg[20] if cfg else 0.7, description="Depth Strength")
depth_settings = widgets.VBox([depth_map_toggle])

depthmap_popup({"new": depth_map_toggle.value})
depth_upload.observe(depthmap_upload_handler, names="value")

openpose_upload = widgets.FileUpload(accept="image/*", multiple=False)
openpose_link_widget = widgets.Text(value=cfg[21] if cfg and (not cfg[21].startswith("/content/openpose/") or os.path.exists("/content/openpose/")) else "", description="OpenPose Link", placeholder="Image link")

openpose_dropdown = widgets.Dropdown(options=controlnet_dropdown_choice, value=controlnet_dropdown_choice[2] if openpose_link_widget.value == "" else controlnet_dropdown_choice[3] if openpose_link_widget.value == "controlnet" else controlnet_dropdown_choice[4] if openpose_link_widget.value == "inpaint" else controlnet_dropdown_choice[0], disabled=False, description="Reference Image")
openpose_toggle = widgets.Checkbox(value=cfg[22] if cfg else False, description="Enable OpenPose")
openpose_strength_slider = widgets.FloatSlider(min=0.1, max=1, step=0.1, value=cfg[23] if cfg else 0.7, description="OpenPose Strength")
openpose_settings = widgets.VBox([openpose_toggle])

openpose_popup({"new": openpose_toggle.value})
openpose_upload.observe(openpose_upload_handler, names="value")

canny_dropdown_handler({"new": canny_dropdown.value})
depthmap_dropdown_handler({"new": depthmap_dropdown.value})
openpose_dropdown_handler({"new": openpose_dropdown.value})

canny_dropdown.observe(canny_dropdown_handler, names="value")
depthmap_dropdown.observe(depthmap_dropdown_handler, names="value")
openpose_dropdown.observe(openpose_dropdown_handler, names="value")

canny_toggle.observe(canny_popup, names="value")
depth_map_toggle.observe(depthmap_popup, names="value")
openpose_toggle.observe(openpose_popup, names="value")

