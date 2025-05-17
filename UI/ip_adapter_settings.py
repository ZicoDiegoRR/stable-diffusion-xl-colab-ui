from diffusers.utils import load_image, make_image_grid
import ipywidgets as widgets
from io import BytesIO
import math
import os
import re

class IPAdapterLoader:
    def collect_values(self): # Function to collect the values in this class
        return [
            self.ip_image_link_widget.value,
            self.ip_adapter_strength_slider.value,
            self.ip_adapter_dropdown.value
        ]

    def wrap_settings(self): # Function to collect every widget into a vbox
        return self.ip_settings

    def check_if_link(self, value): # Function to check whether the given path is a link or a file
        return value.startswith("https://") or value.startswith("http://") or value.startswith("/content/gdrive/MyDrive") or os.path.exists(f"/content/ip_adapter/{value}")
    
    def path_listdir(self): # Function to list files in /content/ip_adapter directory
        return [os.path.join("/content/ip_adapter/", element) for element in os.listdir("/content/ip_adapter/") if os.path.isfile(os.path.join("/content/ip_adapter/", element))]
            
    def ip_remove_button_on_click(self, path): # Function to remove images from widget and directory
        os.remove(path)
        self.ip_grid_button = self.ip_grid_button_maker(sorted(self.path_listdir()))
        self.ip_settings.children = [self.ip_adapter_dropdown, self.ip_image_link_widget, self.ip_image_upload, self.ip_adapter_strength_slider, self.ip_grid_image_html, self.ip_grid_image, self.ip_grid_button_html, self.ip_grid_button]
        if not self.path_listdir():
            self.ip_adapter_dropdown_popup({"new": "refresh_zero"})

    def ip_grid_button_maker(self, list): # Function to make a grid of images and buttons
        list_grid = widgets.GridspecLayout(math.ceil(len(list)/5), 5) if list else widgets.HTML(value="The IP-Adapter folder is empty. Start uploading to use your own image.")
        buffer = BytesIO()
        loaded_image_for_grid = []
        if list:
            for i in range(math.ceil(len(list)/5)):
                for j in range(5):
                    k = (i*5 + j + 1)
                    list_grid[i, j] = widgets.Button(description=f"Remove image {k}", button_style='danger', layout=widgets.Layout(height='auto', width='auto')) if k <= len(list) else widgets.Button(description="", layout=widgets.Layout(height='auto', width='auto'))
                    path = list[k - 1] if k <= len(list) else ""
                    list_grid[i, j].on_click(lambda b, path=path: self.ip_remove_button_on_click(path)) if k <= len(list) else None
                    loaded_image_for_grid.append(load_image(path)) if k <= len(list) else loaded_image_for_grid.append(load_image("https://huggingface.co/IDK-ab0ut/BFIDIW9W29NFJSKAOAOXDOKERJ29W/resolve/main/placeholder.png"))
            ip_image_grid_maker = make_image_grid([element.resize((1024, 1024)) for element in loaded_image_for_grid], rows=math.ceil(len(list)/5), cols=5)
            ip_image_grid_maker.save(buffer, format = "PNG")
            self.ip_grid_image.value = buffer.getvalue()
        else:
            self.ip_settings.children = [self.ip_adapter_dropdown, self.ip_image_link_widget, self.ip_image_upload, self.ip_adapter_strength_slider]
        return list_grid

    def ip_adapter_dropdown_popup(self, change): # Function to show or hide the widgets
        if change["new"] != "None" and change["new"] != "refresh_zero":
            self.ip_settings.children = [self.ip_adapter_dropdown, self.ip_image_link_widget, self.ip_image_upload, self.ip_adapter_strength_slider] if not os.listdir("/content/ip_adapter/") else [self.ip_adapter_dropdown, self.ip_image_link_widget, self.ip_image_upload, self.ip_adapter_strength_slider, self.ip_grid_image_html, self.ip_grid_image, self.ip_grid_button_html, self.ip_grid_button]
        elif change["new"] == "refresh_zero":
            self.ip_settings.children = [self.ip_adapter_dropdown, self.ip_image_link_widget, self.ip_image_upload, self.ip_adapter_strength_slider]
        else:
            self.ip_settings.children = [self.ip_adapter_dropdown]

    def ip_adapter_upload_handler(self, change): # Function to trigger the UI logic when images are uploaded
        for filename, file_info in self.ip_image_upload.value.items():
            with open(f"/content/ip_adapter/{filename}", "wb") as up:
                up.write(file_info["content"])
            self.collected_uploaded_ip_image += f"/content/ip_adapter/{filename},"
        self.ip_grid_button = self.ip_grid_button_maker(sorted(self.path_listdir()))
        self.ip_settings.children = [self.ip_adapter_dropdown, self.ip_image_link_widget, self.ip_image_upload, self.ip_adapter_strength_slider, self.ip_grid_image_html, self.ip_grid_image, self.ip_grid_button_html, self.ip_grid_button]

    def __init__(self, cfg):
        os.makedirs("/content/ip_adapter", exist_ok=True)
        self.collected_uploaded_ip_image = ""
        initial_ip_image = [word for word in re.split(r"\s*,\s*", cfg[0]) if word] if cfg else ""
        filtered_ip_image_during_initial_load = []
        for link in initial_ip_image:
            if self.check_if_link(link):
                filtered_ip_image_during_initial_load.append(link)

        self.ip_grid_image_html = widgets.HTML(value="Uploaded image(s):")
        self.ip_grid_image = widgets.Image()
        self.ip_grid_button_html = widgets.HTML(value="Remove image(s):")
        self.ip_grid_button = self.ip_grid_button_maker(sorted(self.path_listdir)) if self.path_listdir() else widgets.GridspecLayout(1, 5)

        self.ip_image_upload = widgets.FileUpload(accept="image/*", multiple=True)
        self.ip_image_link_widget = widgets.Text(value=",".join(filtered_ip_image_during_initial_load), description="IP Image Link", placeholder="Image links separated by commas")
        self.ip_adapter_strength_slider = widgets.FloatSlider(min=0.1, max=1, step=0.1, value=cfg[1] if cfg else 0.8, description="Adapter Strength")

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
        if not self.ip_image_link_widget.value:
            self.ip_adapter_dropdown.value = "None"

        self.ip_adapter_dropdown_popup({"new": self.ip_adapter_dropdown.value})
        self.ip_image_upload.observe(self.ip_adapter_upload_handler, names="value")
        self.ip_adapter_dropdown.observe(self.ip_adapter_dropdown_popup, names="value")
