from diffusers.utils import load_image, make_image_grid
import ipywidgets as widgets
from io import BytesIO
import math
import os
import re

class IPAdapterLoader:
    # Function to collect the values in this class
    def collect_values(self):
        return [
            self.ip_image_link_widget.value  + ",".join(self.path_listdir()),
            self.ip_adapter_strength_slider.value,
            self.ip_adapter_dropdown.value
        ]

    # Function to return every widget for IP-Adapter
    def return_widgets(self):
        return [
            self.ip_image_link_widget,
            self.ip_adapter_strength_slider,
            self.ip_adapter_dropdown
        ]

    # Function to wrap every widget into a VBox
    def wrap_settings(self): # Function to collect every widget into a vbox
        return self.ip_settings

    # Function to check whether the given path is a link or a file
    def check_if_link(self, value): 
        return value.startswith(("https://", "http://", "/content/gdrive/MyDrive")) or os.path.exists(value)

    # Function to list files in /content/ip_adapter directory
    def path_listdir(self): 
        return [os.path.join("/content/ip_adapter/", element) for element in os.listdir("/content/ip_adapter/") if os.path.isfile(os.path.join("/content/ip_adapter/", element))]

    # Function to remove invalid URLs from the link widget
    def sanitize_links(self, path):
        new_value = re.split(r"\s*,\s*", self.ip_image_link_widget.value)
        new_value.remove(path)
        self.ip_image_link_widget.value = ", ".join(new_value)

    # Function to remove images from widget and directory
    def ip_remove_button_on_click(self, path, type, length): 
        if type == "Upload":
            os.remove(path)
        elif type == "Link":
            self.sanitize_links(path)
            
        self.ip_grid_button = self.ip_grid_button_maker(sorted(self.path_listdir()))
        if length == 1:
            self.ip_adapter_dropdown_popup({"new": "refresh_zero"})
        else:
            self.ip_adapter_dropdown_popup({"new": self.ip_adapter_dropdown.value})

    # Function to make a grid of images and buttons
    def ip_grid_button_maker(self, image_list): 
        list_grid = widgets.GridspecLayout(math.ceil(len(image_list)/5), 5) if image_list else widgets.HTML(value="The IP-Adapter folder is empty. Start uploading to use your own images.")
        buffer = BytesIO()
        loaded_image_for_grid = []
        if image_list:
            for i in range(math.ceil(len(image_list)/5)):
                for j in range(5):
                    k = (i*5 + j + 1)
                    list_grid[i, j] = widgets.Button(description=f"Remove image {k}", button_style='danger', layout=widgets.Layout(height='auto', width='auto')) if k <= len(image_list) else widgets.Button(description="", layout=widgets.Layout(height='auto', width='auto'))
                    path = image_list[k - 1] if k <= len(image_list) else ""
        
                    if path.startswith(("https://", "http://", "/content/gdrive/MyDrive")):
                        image_type = "Link"
                    else:
                        image_type = "Upload"
                        
                    list_grid[i, j].on_click(lambda b, path=path: self.ip_remove_button_on_click(path, image_type, k)) if k <= len(image_list) else None
                    loaded_image_for_grid.append(load_image(path)) if k <= len(image_list) else loaded_image_for_grid.append(load_image("https://huggingface.co/IDK-ab0ut/BFIDIW9W29NFJSKAOAOXDOKERJ29W/resolve/main/placeholder.png"))
                
            ip_image_grid_maker = make_image_grid([element.resize((1024, 1024)) for element in loaded_image_for_grid], rows=math.ceil(len(image_list)/5), cols=5)
            ip_image_grid_maker.save(buffer, format = "PNG")
            self.ip_grid_image.value = buffer.getvalue()
        else:
            self.ip_adapter_dropdown_popup({"new": "refresh_zero"})
        return list_grid

    # Function to show or hide the widgets
    def ip_adapter_dropdown_popup(self, change):
        if change["new"] != "None" and change["new"] != "refresh_zero":
            self.ip_settings.children = [
                self.ip_adapter_dropdown, 
                self.ip_image_link_widget, 
                self.ip_image_upload, 
                self.ip_adapter_strength_slider,
                self.ip_adapter_preview_button,
            ] if not self.path_listdir() and not self.ip_image_link_widget.value else [
                self.ip_adapter_dropdown, 
                self.ip_image_link_widget, 
                self.ip_image_upload, 
                self.ip_adapter_strength_slider, 
                self.ip_adapter_preview_button,
                self.ip_grid_image_html, 
                self.ip_grid_image, 
                self.ip_grid_button_html, 
                self.ip_grid_button,
            ]
            
        elif change["new"] == "refresh_zero":
            self.ip_settings.children = [
                self.ip_adapter_dropdown, 
                self.ip_image_link_widget, 
                self.ip_image_upload, 
                self.ip_adapter_strength_slider,
                self.ip_adapter_preview_button,
            ]
        else:
            self.ip_settings.children = [self.ip_adapter_dropdown]

    # Function to save uploaded images locally
    def ip_adapter_upload_handler(self, change): 
        for filename, file_info in self.ip_image_upload.value.items():
            with open(f"/content/ip_adapter/{filename}", "wb") as up:
                up.write(file_info["content"])

    # Function to handle the preview
    def preview_grid(self):
        links_from_widget = [
            word for word in re.split(r"\s*,\s*", self.ip_image_link_widget.value) if word and self.check_if_link(word)
        ] if self.ip_image_link_widget else []
        combined_images = links_from_widget + sorted(self.path_listdir())

        sanitized_img = []
        for img in combined_images:
            try:
                _ = load_image(img)
                sanitized_img.append(img)
            except Exception as e:
                if not img.startswith("/content/ip_adapter/"):
                    self.sanitize_links(img)
                else:
                    os.remove(img)

        self.ip_grid_button = self.ip_grid_button_maker(sanitized_img)
        self.ip_adapter_dropdown_popup({"new": self.ip_adapter_dropdown.value})

    # Initialize everything
    def __init__(self, cfg):
        os.makedirs("/content/ip_adapter", exist_ok=True)
        filtered_ip_image_during_initial_load = [word for word in re.split(r"\s*,\s*", cfg[0]) if word and self.check_if_link(word)] if cfg else ""
        
        self.ip_grid_image_html = widgets.HTML(value="Uploaded image(s):")
        self.ip_grid_image = widgets.Image()
        self.ip_grid_button_html = widgets.HTML(value="Remove image(s):")
        self.ip_grid_button = None

        self.ip_image_upload = widgets.FileUpload(accept="image/*", multiple=True)
        self.ip_image_link_widget = widgets.Text(value=", ".join(filtered_ip_image_during_initial_load), description="IP Image Link", placeholder="Image links separated by commas")
        self.ip_adapter_strength_slider = widgets.FloatSlider(min=0.1, max=1, step=0.1, value=cfg[1] if cfg else 0.8, description="Adapter Strength")
        self.ip_adapter_preview_button = widgets.Button(description="Preview")

        self.ip_adapter_dropdown = widgets.Dropdown(
            options=[
                "ip-adapter-plus_sdxl_vit-h.bin",
                "ip-adapter-plus-face_sdxl_vit-h.bin",
                "ip-adapter_sdxl_vit-h.bin",
                "None"
            ],
            value=cfg[2] if cfg else "None",
            description="IP-Adapter",
        )
        
        self.ip_settings = widgets.VBox()

        self.ip_adapter_dropdown_popup({"new": self.ip_adapter_dropdown.value})
        self.ip_image_upload.observe(self.ip_adapter_upload_handler, names="value")
        self.ip_adapter_dropdown.observe(self.ip_adapter_dropdown_popup, names="value")
        self.ip_adapter_preview_button.on_click(lambda b: self.preview_grid())
