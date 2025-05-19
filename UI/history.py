import ipywidgets as widgets
import time
import math
import os
import re

class HistorySystem:
    def wrap_settings(self): # Function to collect every widget into a vbox
        return self.history_display_vbox

    def list_images(self, path): # Function to list every image in a folder
        if os.path.exists(path):
            return sorted([os.path.join(path, element) for element in os.listdir(path) if element.endswith(".png") and os.path.isfile(os.path.join(path, element))], key=os.path.getmtime, reverse=True)
        else:
            return []
        
    def history_quick_reference_controlnet_selector(self, type, path, controlnet, tab): # Function to input the image as the reference for ControlNet
        ui = tab.ui
        if type == "canny":
            controlnet.canny_link_widget.value = path
            controlnet.canny_dropdown.value = "Link"
            controlnet.canny_toggle.value = True
            ui.selected_index = 2
        elif type == "depthmap":
            controlnet.depth_map_link_widget.value = path
            controlnet.depthmap_dropdown.value = "Link"
            controlnet.depth_map_toggle.value = True
            ui.selected_index = 2
        elif type == "openpose":
            controlnet.openpose_link_widget.value = path
            controlnet.openpose_dropdown.value = "Link"
            controlnet.openpose_toggle.value = True
            ui.selected_index = 2
        self.history_button_handler(path)

    def history_quick_reference_second(self, type, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab): # Function to input the image as the reference image
        ui = tab.ui
        if type == "img2img":
            img2img.reference_image_link_widget.value = path
            self.history_button_handler(path)
            ui.selected_index = 1
        elif type == "inpainting":
            inpaint.inpainting_image_dropdown.value = path
            self.history_button_handler(path)
            inpaint.inpainting_toggle.value = True
            ui.selected_index = 3
        elif type == "ip":
            if ip_adapter_dropdown.value == "None":
              ip.ip_adapter_dropdown.value = "ip-adapter_sdxl_vit-h.bin"
            if ip_image_link_widget.value == "":
              ip.ip_image_link_widget.value = path
            else:
              ip.ip_image_link_widget.value += "," + path
            self.history_button_handler(path)
            ui.selected_index = 6
        elif type == "controlnet":
            self.history_image_display_first.children = [widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), self.history_image_widget, self.history_image_modification_date, widgets.HBox([self.history_quick_reference_canny, self.history_quick_reference_depthmap, self.history_quick_reference_openpose]), self.history_back_button_second]

            self.history_back_button_second._click_handlers.callbacks.clear()
            self.history_quick_reference_canny._click_handlers.callbacks.clear()
            self.history_quick_reference_depthmap._click_handlers.callbacks.clear()
            self.history_quick_reference_openpose._click_handlers.callbacks.clear()

            self.history_quick_reference_canny.on_click(lambda b: self.history_quick_reference_controlnet_selector("canny", path, controlnet, tab))
            self.history_quick_reference_depthmap.on_click(lambda b: self.history_quick_reference_controlnet_selector("depthmap", path, controlnet, tab))
            self.history_quick_reference_openpose.on_click(lambda b: self.history_quick_reference_controlnet_selector("openpose", path, controlnet, tab))
            self.history_back_button_second.on_click(lambda b: self.history_quick_reference_first(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))
        elif type == "upscale":
            upscaler.upscale_widget.input_link.value = path
            self.history_button_handler(path)
            ui.selected_index = 7

    def history_quick_reference_first(self, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab): # Function to use an image from history to be the reference image of Img2Img, ControlNet, or Inpainting
        self.history_image_display_first.children = [widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), self.history_image_widget, self.history_image_modification_date, widgets.HBox([self.history_quick_reference_img2img, self.history_quick_reference_controlnet, self.history_quick_reference_inpainting, self.history_quick_reference_ip_adapter, self.history_quick_reference_upscale]), self.history_back_button_first]

        self.history_back_button_first._click_handlers.callbacks.clear()
        self.history_quick_reference_img2img._click_handlers.callbacks.clear()
        self.history_quick_reference_controlnet._click_handlers.callbacks.clear()
        self.history_quick_reference_inpainting._click_handlers.callbacks.clear()
        self.history_quick_reference_ip_adapter._click_handlers.callbacks.clear()
        self.history_quick_reference_upscale._click_handlers.callbacks.clear()

        self.history_back_button_first.on_click(lambda b: self.history_button_handler(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))
        self.history_quick_reference_img2img.on_click(lambda b: self.history_quick_reference_second("img2img", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))
        self.history_quick_reference_controlnet.on_click(lambda b: self.history_quick_reference_second("controlnet", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))
        self.history_quick_reference_inpainting.on_click(lambda b: self.history_quick_reference_second("inpainting", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))
        self.history_quick_reference_ip_adapter.on_click(lambda b: self.history_quick_reference_second("ip", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))
        self.history_quick_reference_upscale.on_click(lambda b: self.history_quick_reference_second("upscale", path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))

    def history_button_handler(self, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab): # Function to show and replace image from history upon clicking a button
        try:
            self.history_image_widget.value = open(path, "rb").read()
            self.history_image_modification_date.value = f"Last modification time: {time.strftime('%B, %d %Y %H:%M:%S', time.localtime(os.path.getmtime(path)))}"
        except Exception as e:
            self.history_image_modification_date.value = f"An error occured when trying to read the image. Reason: {e}"
        self.history_image_display_first.children = [widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), self.history_image_widget, self.history_image_modification_date, self.history_quick_reference_button]

        self.history_quick_reference_button._click_handlers.callbacks.clear()
        self.history_quick_reference_button.on_click(lambda b: self.history_quick_reference_first(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab))

    def grid(self, list, path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab): # Function to make a grid of buttons
        list_grid = widgets.GridspecLayout(math.ceil(len(list)/10), 10) if list else widgets.HTML(value="Nothing in here currently.")
        if list:
            for i in range(math.ceil(len(list)/10)):
                for j in range(10):
                    k = (i*10 + j + 1)
                    list_grid[i, j] = widgets.Button(description=str(k), layout=widgets.Layout(height='auto', width='auto')) if k <= len(list) else widgets.Button(description="", layout=widgets.Layout(height='auto', width='auto'))
                    path = list[k - 1] if k <= len(list) else ""
                    list_grid[i, j].on_click(lambda b, path=path: self.history_button_handler(path, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab)) if k <= len(list) else None
        return list_grid

    def history_display(self, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab): # Main logic for history
        base_path = "/content/gdrive/MyDrive" if os.path.exists("/content/gdrive/MyDrive") else "/content"
        text2img_listdir = self.list_images(f"{base_path}/Text2Img")
        controlnet_listdir = self.list_images(f"{base_path}/ControlNet")
        inpainting_listdir = self.list_images(f"{base_path}/Inpainting")
        img2img_listdir = self.list_images(f"{base_path}/Img2Img")
        upscale_listdir = self.list_images(f"{base_path}/Upscaled")

        text2img_list = self.grid(text2img_listdir, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab)
        controlnet_list = self.grid(controlnet_listdir, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab)
        inpainting_list = self.grid(inpainting_listdir, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab)
        img2img_list = self.grid(img2img_listdir, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab)
        upscale_list = self.grid(upscale_listdir, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab)

        self.history_accordion = widgets.Accordion(continuous_update = True)
        self.history_image_modification_date = widgets.HTML()
        self.history_image_widget = widgets.Image()

        self.history_accordion.children = [text2img_list, img2img_list, controlnet_list, inpainting_list, upscale_list]
        return text2img_list, controlnet_list, inpainting_list, img2img_list, upscale_list

    def __init__(self, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, upscaler, tab):
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

        self.history_image_display_first = widgets.VBox([widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), self.history_image_widget, self.history_image_modification_date], continuous_update = True)
        text2img_list, controlnet_list, inpainting_list, img2img_list, upscale_list = self.history_display()
        self.history_accordion.children = [text2img_list, img2img_list, controlnet_list, inpainting_list, upscale_list]

        history_accordion_titles = ["Text-to-Image History 🔮✍", "Image-to-Image History 🔮🎨", "ControlNet History 🔮🔧", "Inpainting History 🔮🖌️", "Image Upscaler History 🔮✨"]
        for i, title in enumerate(history_accordion_titles):
            self.history_accordion.set_title(i, title)

        self.history_display_vbox = widgets.VBox([self.history_accordion, self.history_image_display_first], continuous_update = True)
