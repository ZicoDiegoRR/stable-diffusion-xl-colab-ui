import ipywidgets as widgets

class InpaintingSettings:
    def return_widgets(self):
        return [
            self.inpainting_image_dropdown,
            self.mask_image_widget,
            self.inpainting_toggle,
            self.inpainting_strength_slider,
        ]
        
    def wrap_settings(self):
        return [
            widgets.HTML(value="<b>To be updated in the future.</b>"),
            self.inpainting_image_dropdown,
            self.mask_image_widget,
            self.inpainting_toggle,
            self.inpainting_strength_slider,
        ]

    def collect_values(self):
        return [
            self.inpainting_image_dropdown.value,
            self.mask_image_widget.value,
            self.inpainting_toggle.value,
            self.inpainting_strength_slider.value,
        ]
    def __init__(self, cfg):
        self.inpainting_image_dropdown = widgets.Combobox(
            options=[
                "pre-generated text2image image",
                "pre-generated controlnet image",
                "previous inpainting image"
            ],
            value=cfg[0] if cfg else "pre-generated text2image image",
            description="Inpainting Image",
            ensure_option=False
        )
        self.mask_image_widget = widgets.Text(value=cfg[1] if cfg else "", description="Mask Image", placeholder="Image link")
        self.inpainting_toggle = widgets.Checkbox(value=cfg[2] if cfg else False, description="Enable Inpainting")
        self.inpainting_strength_slider = widgets.FloatSlider(min=0.1, max=1, step=0.01, value=cfg[3] if cfg else 0.9, description="Inpainting Strength")
