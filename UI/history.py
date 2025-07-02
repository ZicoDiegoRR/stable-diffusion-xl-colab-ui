import ipywidgets as widgets
from PIL import Image
import time
import math
import json
import os
import re

# Function to save the last-generated images
def save_last(base_path, text2img, controlnet, inpaint, img2img):
    with open(f"{base_path}/Saved Parameters/last_generation.json", "w") as f:
        image_dict = {
            "text2img": text2img[0] if text2img else None,
            "controlnet": controlnet[0] if controlnet else None,
            "inpaint": inpaint[0] if inpaint else None,
            "img2img": img2img[0] if img2img else None,
        }
        json.dump(image_dict, f, indent=4)

# Function to check if the file is an image
def img_check(path):
    return path.endswith((".png", ".jfif", ".jpe", ".jpg", ".jpeg", ".webp", ".ico", ".bmp"))

class HistorySystem:
    # Function to collect every widget into a vbox
    def wrap_settings(self):
        return self.history_display_vbox

    # Function to list every image in a folder
    def list_images(self, path): 
        if os.path.exists(path):
            return sorted(
                [os.path.join(path, element) for element in os.listdir(path) if img_check(element) and os.path.isfile(os.path.join(path, element))], 
                key=os.path.getmtime, reverse=True
            )
        else:
            return []

    # Assign the widgets to the accordion
    def assign_children(self):
        self.history_accordion.children = [
            self.text2img_grid, 
            self.img2img_grid,
            self.controlnet_grid, 
            self.inpainting_grid,  
            self.upscale_grid,
        ]

    # Function to input the image as the reference for ControlNet
    def history_quick_reference_controlnet_selector(self, type, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path):
        if type == "canny":
            controlnet.canny_link_widget.value = path
            controlnet.canny_dropdown.value = "Link"
            controlnet.canny_toggle.value = True
            tab.selected_index = 2
        elif type == "depthmap":
            controlnet.depth_map_link_widget.value = path
            controlnet.depthmap_dropdown.value = "Link"
            controlnet.depth_map_toggle.value = True
            tab.selected_index = 2
        elif type == "openpose":
            controlnet.openpose_link_widget.value = path
            controlnet.openpose_dropdown.value = "Link"
            controlnet.openpose_toggle.value = True
            tab.selected_index = 2
        self.history_button_handler(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path)

    # Function to input the image as the reference image
    def history_quick_reference_second(self, type, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path): 
        if type == "img2img":
            img2img.reference_image_link_widget.value = path
            self.history_button_handler(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path)
            tab.selected_index = 1
        elif type == "inpainting":
            inpaint.inpainting_image_dropdown.value = path
            self.history_button_handler(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path)
            inpaint.inpainting_toggle.value = True
            tab.selected_index = 3
        elif type == "ip":
            if ip.ip_adapter_dropdown.value == "None":
              ip.ip_adapter_dropdown.value = "ip-adapter_sdxl_vit-h.bin"
            if ip.ip_image_link_widget.value == "":
              ip.ip_image_link_widget.value = path
            else:
              ip.ip_image_link_widget.value += "," + path
            self.history_button_handler(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path)
            tab.selected_index = 6
        elif type == "controlnet":
            self.history_image_display_first.children = [
                widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), 
                self.history_image_widget, self.history_image_modification_date, self.resolution,
                widgets.HBox([
                    self.history_quick_reference_canny, 
                    self.history_quick_reference_depthmap,
                    self.history_quick_reference_openpose
                ]), 
                self.history_back_button_second
            ]

            self.history_back_button_second._click_handlers.callbacks.clear()
            self.history_quick_reference_canny._click_handlers.callbacks.clear()
            self.history_quick_reference_depthmap._click_handlers.callbacks.clear()
            self.history_quick_reference_openpose._click_handlers.callbacks.clear()

            self.history_quick_reference_canny.on_click(lambda b: self.history_quick_reference_controlnet_selector(
                "canny", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
            ))
            self.history_quick_reference_depthmap.on_click(lambda b: self.history_quick_reference_controlnet_selector(
                "depthmap", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
            ))
            self.history_quick_reference_openpose.on_click(lambda b: self.history_quick_reference_controlnet_selector(
                "openpose", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
            ))
            self.history_back_button_second.on_click(lambda b: self.history_quick_reference_first(
                path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
            ))
        elif type == "upscale":
            upscaler.input_link.value = path
            self.history_button_handler(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path)
            tab.selected_index = 7

    # Function to use an image from history to be the reference image of Img2Img, ControlNet, or Inpainting
    def history_quick_reference_first(self, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path): 
        self.history_image_display_first.children = [
            widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), 
            self.history_image_widget, 
            self.history_image_modification_date, 
            self.resolution, 
            widgets.HBox([
                self.history_quick_reference_img2img, 
                self.history_quick_reference_controlnet, 
                self.history_quick_reference_inpainting, 
                self.history_quick_reference_ip_adapter, 
                self.history_quick_reference_upscale
            ]), 
            self.history_back_button_first
        ]

        self.history_back_button_first._click_handlers.callbacks.clear()
        self.history_quick_reference_img2img._click_handlers.callbacks.clear()
        self.history_quick_reference_controlnet._click_handlers.callbacks.clear()
        self.history_quick_reference_inpainting._click_handlers.callbacks.clear()
        self.history_quick_reference_ip_adapter._click_handlers.callbacks.clear()
        self.history_quick_reference_upscale._click_handlers.callbacks.clear()

        self.history_back_button_first.on_click(lambda b: self.history_button_handler(
            path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
        ))
        self.history_quick_reference_img2img.on_click(lambda b: self.history_quick_reference_second(
            "img2img", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
        ))
        self.history_quick_reference_controlnet.on_click(lambda b: self.history_quick_reference_second(
            "controlnet", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
        ))
        self.history_quick_reference_inpainting.on_click(lambda b: self.history_quick_reference_second(
            "inpainting", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
        ))
        self.history_quick_reference_ip_adapter.on_click(lambda b: self.history_quick_reference_second(
            "ip", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
        ))
        self.history_quick_reference_upscale.on_click(lambda b: self.history_quick_reference_second(
            "upscale", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
        ))

    # Function to show and replace image from history upon clicking a button
    def history_button_handler(self, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path): 
        try:
            image = open(path, "rb")
            self.history_image_widget.value = image.read()
            modification_time = time.strftime('%B, %d %Y %H:%M:%S', time.localtime(os.path.getmtime(path)))

            width, height = Image.open(path).size
            self.history_image_modification_date.value = f"Last modification time: {modification_time}"
            self.resolution.value = f"Resolution: {width}x{height}"
            
            self.delete_button_after_click._click_handlers.callbacks.clear()
            self.delete_button_after_click.on_click(lambda b: self.history_delete_handler(
                path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path,
            ))
            
        except Exception as e:
            self.history_image_modification_date.value = f"An error occured when trying to read the image. Reason: {e}"

        else:
            self.history_image_display_first.children = [
                widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), 
                self.history_image_widget, self.history_image_modification_date, self.resolution,
                widgets.HBox([self.history_quick_reference_button, self.delete_button_after_click]),
            ]

            self.history_quick_reference_button._click_handlers.callbacks.clear()
            self.history_quick_reference_button.on_click(lambda b: self.history_quick_reference_first(
                path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
            ))

    # Function to delete an image from the history
    def history_delete_handler(self, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path):
        os.remove(path)
        self.history_image_widget.value = b''
        self.history_image_modification_date.value = ""
        self.history_image_display_first.children = [
            widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), 
            self.history_image_widget,
            self.history_image_modification_date, 
        ]
        self.history_update(text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path)
        
    # Function to show the pages
    def arrow_handler(self, list_path, text2img, img2img, controlnet, inpaint, ip,
                       lora, embeddings, upscaler, tab, page_index, history_type, change
    ):
        index = page_index + change
        if history_type == "text2img":
            self.text2img_page_index = index
            self.text2img_grid = self.grid(
                list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type
            )
        elif history_type == "controlnet":
            self.controlnet_page_index = index
            self.controlnet_grid = self.grid(
                list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type
            )
        elif history_type == "inpaint":
            self.inpainting_page_index = index
            self.inpainting_grid = self.grid(
                list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type
            )
        elif history_type == "img2img":
            self.img2img_page_index = index
            self.img2img_grid = self.grid(
                list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type
            )
        elif history_type == "upscale":
            self.upscale_page_index = index
            self.upscale_grid = self.grid(
                list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type
            )

        self.assign_children()

    # Function to make a grid of buttons
    def grid(self, list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type, base_path):
        if list_path:
            pages_amount = int(math.ceil(len(list_path)/50))
            if (index + 1) != pages_amount or len(list_path) % 50 == 0:
                max_buttons = 50 
            else: 
                max_buttons = len(list_path) % 50
            row = math.ceil(max_buttons/10)
            grid = widgets.GridspecLayout(n_rows=row, n_columns=10)
            for i in range(row):
                for j in range(10):
                    k = (index*50 + i*10 + j + 1)
                    grid[i, j] = widgets.Button(
                        description=str(k), layout=widgets.Layout(height='auto', width='auto')
                    ) if k <= len(list_path) else widgets.Button(
                        description="", layout=widgets.Layout(height='auto', width='auto')
                    )
                    path = list_path[k - 1] if k <= len(list_path) else ""
                    grid[i, j].on_click(lambda b, path=path: self.history_button_handler(
                        path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path
                    )) if k <= len(list_path) else None
                    
            page_mark = widgets.Label(value=f"Page {index + 1} of {pages_amount}")
            total_image = widgets.Label(value=f"Total images: {len(list_path)}", layout=widgets.Layout(margin='0 0 0 auto'))

            previous_button = widgets.Button(
                description="←", 
                layout=widgets.Layout(width="50%"), 
                disabled=bool(index == 0)
            )
            next_button = widgets.Button(
                description="→", 
                layout=widgets.Layout(width="50%"), 
                disabled=bool((index + 1) == pages_amount)
            )

            previous_button.on_click(lambda b: self.arrow_handler(
                list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type, -1
            ))
            next_button.on_click(lambda b: self.arrow_handler(
                list_path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, index, history_type, 1
            ))
            
            list_grid = widgets.VBox([
                widgets.HBox([
                    page_mark, total_image,
                ]),
                grid,
                widgets.Label(" "),
                widgets.HBox([
                    previous_button, next_button,
                ]),
            ])
            
        else:
            list_grid = widgets.HTML(value="The history is empty. Start generating to display images.")

        return list_grid

    # Main logic for history
    def history_display(self, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path, grid, index=0):
        if not grid:
            text2img_list = self.list_images(f"{base_path}/Text2Img")
            controlnet_list = self.list_images(f"{base_path}/ControlNet")
            inpainting_list = self.list_images(f"{base_path}/Inpainting")
            img2img_list = self.list_images(f"{base_path}/Img2Img")
            upscale_list = self.list_images(f"{base_path}/Upscaled")

        elif grid:
            text2img_list = self.grid(
                self.text2img_listdir, text2img, img2img, controlnet, inpaint, 
                ip, lora, embeddings, upscaler, tab, self.text2img_page_index,
                "text2img", base_path,
            )
            controlnet_list = self.grid(
                self.controlnet_listdir, text2img, img2img, controlnet, inpaint, 
                ip, lora, embeddings, upscaler, tab, self.controlnet_page_index,
                "controlnet", base_path,
            )
            inpainting_list = self.grid(
                self.inpainting_listdir, text2img, img2img, controlnet, inpaint, 
                ip, lora, embeddings, upscaler, tab, self.inpainting_page_index,
                "inpaint", base_path,
            )
            img2img_list = self.grid(
                self.img2img_listdir, text2img, img2img, controlnet, inpaint, 
                ip, lora, embeddings, upscaler, tab, self.img2img_page_index,
                "img2img", base_path,
            )
            upscale_list = self.grid(
                self.upscale_listdir, text2img, img2img, controlnet, inpaint, 
                ip, lora, embeddings, upscaler, tab, self.upscale_page_index,
                "upscale", base_path,
            )

        return text2img_list, controlnet_list, inpainting_list, img2img_list, upscale_list

    # Function to update the history
    def history_update(self, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path):
        self.text2img_listdir, self.controlnet_listdir, self.inpainting_listdir, self.img2img_listdir, self.upscale_listdir = self.history_display(
            text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path, grid=False
        )
        self.text2img_grid, self.controlnet_grid, self.inpainting_grid, self.img2img_grid, self.upscale_grid = self.history_display(
            text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path, grid=True
        )
        self.assign_children()

        save_last(base_path, self.text2img_listdir, self.controlnet_listdir, self.inpainting_listdir, self.img2img_listdir)

    # Initialize
    def __init__(self, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path):
        self.history_quick_reference_button = widgets.Button(description="Use as a reference")
        self.history_quick_reference_img2img = widgets.Button(description="Image-to-image")
        self.history_quick_reference_controlnet = widgets.Button(description="ControlNet")
        self.history_quick_reference_inpainting = widgets.Button(description="Inpainting")
        self.history_quick_reference_ip_adapter = widgets.Button(description="IP-Adapter")
        self.history_quick_reference_upscale = widgets.Button(description="Upscale")
        self.history_back_button_first = widgets.Button(description="Back", button_style='danger')
        self.history_back_button_second = widgets.Button(description="Back", button_style='danger')

        self.history_quick_reference_canny = widgets.Button(description="Canny")
        self.history_quick_reference_depthmap = widgets.Button(description="DepthMap")
        self.history_quick_reference_openpose = widgets.Button(description="OpenPose")

        self.history_accordion = widgets.Accordion(continuous_update = True)
        self.history_image_modification_date = widgets.HTML()
        self.history_image_widget = widgets.Image()
        self.resolution = widgets.Label()

        self.history_image_display_first = widgets.VBox([
            widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), self.history_image_widget, 
            self.history_image_modification_date
        ], continuous_update = True)
        self.delete_button_after_click = widgets.Button(
                description="Delete", 
                button_style="danger",
        )

        self.text2img_page_index = 0
        self.controlnet_page_index = 0
        self.inpainting_page_index = 0
        self.img2img_page_index = 0
        self.upscale_page_index = 0
        
        self.history_update(text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab, base_path)
        
        self.assign_children()

        history_accordion_titles = [
            "Text-to-Image History 🔮✍", 
            "Image-to-Image History 🔮🎨", 
            "ControlNet History 🔮🔧", 
            "Inpainting History 🔮🖌️", 
            "Image Upscaler History 🔮✨"
        ]
        for i, title in enumerate(history_accordion_titles):
            self.history_accordion.set_title(i, title)

        self.history_display_vbox = widgets.VBox([self.history_accordion, self.history_image_display_first])
