import ipywidgets as widgets
import os
import re

def history_quick_reference_controlnet_selector(type, path): # Function to input the image as the reference for ControlNet
  global canny_link_widget, canny_toggle, canny_settings, depth_map_link_widget, depth_map_toggle, depth_settings, openpose_link_widget, openpose_toggle, openpose_settings
  if type == "canny":
    canny_link_widget.value = path
    canny_dropdown.value = "Link"
    canny_toggle.value = True
    ui.selected_index = 2
    canny_settings.children = [canny_toggle, canny_dropdown, canny_link_widget, canny_min_slider, canny_max_slider, canny_strength_slider]
  elif type == "depthmap":
    depth_map_link_widget.value = path
    depthmap_dropdown.value = "Link"
    depth_map_toggle.value = True
    ui.selected_index = 2
    depth_settings.children = [depth_map_toggle, depthmap_dropdown, depth_map_link_widget, depth_strength_slider]
  elif type == "openpose":
    openpose_link_widget.value = path
    openpose_dropdown.value = "Link"
    openpose_toggle.value = True
    ui.selected_index = 2
    openpose_settings.children = [openpose_toggle, openpose_dropdown, openpose_link_widget, openpose_strength_slider]
  history_button_handler(path)

def history_quick_reference_second(type, path): # Function to input the image as the reference image
  global ui, ip_adapter_dropdown, reference_image_link_widget, inpainting_image_dropdown, inpainting_toggle
  if type == "img2img":
    reference_image_link_widget.value = path
    history_button_handler(path)
    ui.selected_index = 1
  elif type == "inpainting":
    inpainting_image_dropdown.value = path
    history_button_handler(path)
    inpainting_toggle.value = True
    ui.selected_index = 3
  elif type == "ip":
    if ip_adapter_dropdown.value == "None":
      ip_adapter_dropdown.value = "ip-adapter_sdxl_vit-h.bin"
    if ip_image_link_widget.value == "":
      ip_image_link_widget.value = path
    else:
      ip_image_link_widget.value += "," + path
    history_button_handler(path)
    ui.selected_index = 6
  elif type == "controlnet":
    history_back_button_second = widgets.Button(description="Back", button_style='danger')
    history_image_display_first.children = [widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), history_image_widget, history_image_modification_date, widgets.HBox([history_quick_reference_canny, history_quick_reference_depthmap, history_quick_reference_openpose]), history_back_button_second]

    history_back_button_second._click_handlers.callbacks.clear()
    history_quick_reference_canny._click_handlers.callbacks.clear()
    history_quick_reference_depthmap._click_handlers.callbacks.clear()
    history_quick_reference_openpose._click_handlers.callbacks.clear()

    history_quick_reference_canny.on_click(lambda b: history_quick_reference_controlnet_selector("canny", path))
    history_quick_reference_depthmap.on_click(lambda b: history_quick_reference_controlnet_selector("depthmap", path))
    history_quick_reference_openpose.on_click(lambda b: history_quick_reference_controlnet_selector("openpose", path))
    history_back_button_second.on_click(lambda b: history_quick_reference_first(path))
  elif type == "upscale":
    upscale_widget.input_link.value = path
    history_button_handler(path)
    ui.selected_index = 7

def history_quick_reference_first(path): # Function to use an image from history to be the reference image of Img2Img, ControlNet, or Inpainting
  history_back_button_first = widgets.Button(description="Back", button_style='danger')
  history_image_display_first.children = [widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), history_image_widget, history_image_modification_date, widgets.HBox([history_quick_reference_img2img, history_quick_reference_controlnet, history_quick_reference_inpainting, history_quick_reference_ip_adapter, history_quick_reference_upscale]), history_back_button_first]

  history_back_button_first._click_handlers.callbacks.clear()
  history_quick_reference_img2img._click_handlers.callbacks.clear()
  history_quick_reference_controlnet._click_handlers.callbacks.clear()
  history_quick_reference_inpainting._click_handlers.callbacks.clear()
  history_quick_reference_ip_adapter._click_handlers.callbacks.clear()
  history_quick_reference_upscale._click_handlers.callbacks.clear()

  history_back_button_first.on_click(lambda b: history_button_handler(path))
  history_quick_reference_img2img.on_click(lambda b: history_quick_reference_second("img2img", path))
  history_quick_reference_controlnet.on_click(lambda b: history_quick_reference_second("controlnet", path))
  history_quick_reference_inpainting.on_click(lambda b: history_quick_reference_second("inpainting", path))
  history_quick_reference_ip_adapter.on_click(lambda b: history_quick_reference_second("ip", path))
  history_quick_reference_upscale.on_click(lambda b: history_quick_reference_second("upscale", path))

def history_button_handler(path): # Function to show and replace image from history upon clicking a button
  history_image_widget.value = open(path, "rb").read()
  history_image_modification_date.value = f"Last modification time: {time.strftime('%B, %d %Y %H:%M:%S', time.localtime(os.path.getmtime(path)))}"
  history_image_display_first.children = [widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), history_image_widget, history_image_modification_date, history_quick_reference_button]

  history_quick_reference_button._click_handlers.callbacks.clear()
  history_quick_reference_button.on_click(lambda b: history_quick_reference_first(path))

def grid(list): # Function to make a grid of buttons
  list_grid = widgets.GridspecLayout(math.ceil(len(list)/10), 10) if list else widgets.HTML(value="Nothing in here currently.")
  if list:
    for i in range(math.ceil(len(list)/10)):
      for j in range(10):
        k = (i*10 + j + 1)
        list_grid[i, j] = widgets.Button(description=str(k), layout=widgets.Layout(height='auto', width='auto')) if k <= len(list) else widgets.Button(description="", layout=widgets.Layout(height='auto', width='auto'))
        path = list[k - 1] if k <= len(list) else ""
        list_grid[i, j].on_click(lambda b, path=path: history_button_handler(path)) if k <= len(list) else None
  return list_grid

def history_display(): # Main logic for history
  text2img_listdir = sorted([os.path.join(f"{base_path}/Text2Img", element) for element in os.listdir(f"{base_path}/Text2Img") if element.endswith(".png") and os.path.isfile(os.path.join(f"{base_path}/Text2Img", element))], key=os.path.getmtime, reverse=True) if os.path.exists(f"{base_path}/Text2Img") else []
  controlnet_listdir = sorted([os.path.join(f"{base_path}/ControlNet", element) for element in os.listdir(f"{base_path}/ControlNet") if element.endswith(".png") and os.path.isfile(os.path.join(f"{base_path}/ControlNet", element))], key=os.path.getmtime, reverse=True) if os.path.exists(f"{base_path}/ControlNet") else []
  inpainting_listdir = sorted([os.path.join(f"{base_path}/Inpainting", element) for element in os.listdir(f"{base_path}/Inpainting") if element.endswith(".png") and os.path.isfile(os.path.join(f"{base_path}/Inpainting", element))], key=os.path.getmtime, reverse=True) if os.path.exists(f"{base_path}/Inpainting") else []
  img2img_listdir = sorted([os.path.join(f"{base_path}/Img2Img", element) for element in os.listdir(f"{base_path}/Img2Img") if element.endswith(".png") and os.path.isfile(os.path.join(f"{base_path}/Img2Img", element))], key=os.path.getmtime, reverse=True) if os.path.exists(f"{base_path}/Img2Img") else []
  upscale_listdir = sorted([os.path.join(f"{base_path}/Upscaled", element) for element in os.listdir(f"{base_path}/Upscaled") if os.path.isfile(os.path.join(f"{base_path}/Upscaled", element))], key=os.path.getmtime, reverse=True) if os.path.exists(f"{base_path}/Upscaled") else []

  text2img_list = grid(text2img_listdir)
  controlnet_list = grid(controlnet_listdir)
  inpainting_list = grid(inpainting_listdir)
  img2img_list = grid(img2img_listdir)
  upscale_list = grid(upscale_listdir)

  history_accordion = widgets.Accordion(continuous_update = True)
  history_image_modification_date = widgets.HTML()
  history_image_widget = widgets.Image()

  history_image_display_first = widgets.VBox([widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), history_image_widget, history_image_modification_date])
  history_accordion.children = [text2img_list, img2img_list, controlnet_list, inpainting_list, upscale_list]

  history_accordion.set_title(0, "Text-to-Image History")
  history_accordion.set_title(1, "Image-to-Image History")
  history_accordion.set_title(2, "ControlNet History")
  history_accordion.set_title(3, "Inpainting History")
  history_accordion.set_title(4, "Image Upscaler History")
  history_display_vbox = widgets.VBox([history_accordion, history_image_display_first])
  return text2img_list, controlnet_list, inpainting_list, img2img_list, upscale_list

history_quick_reference_button = widgets.Button(description="Use as a reference")
history_quick_reference_img2img = widgets.Button(description="Image-to-image")
history_quick_reference_controlnet = widgets.Button(description="ControlNet")
history_quick_reference_inpainting = widgets.Button(description="Inpainting")
history_quick_reference_ip_adapter = widgets.Button(description="IP-Adapter")
history_quick_reference_upscale = widgets.Button(description="Upscale")

history_quick_reference_canny = widgets.Button(description="Canny")
history_quick_reference_depthmap = widgets.Button(description="DepthMap")
history_quick_reference_openpose = widgets.Button(description="OpenPose")

history_accordion = widgets.Accordion(continuous_update = True)
history_image_modification_date = widgets.HTML()
history_image_widget = widgets.Image()

history_image_display_first = widgets.VBox([widgets.HTML(value="Image will show up here. (from the newest to the oldest)"), history_image_widget, history_image_modification_date], continuous_update = True)
text2img_list, controlnet_list, inpainting_list, img2img_list, upscale_list = history_display()
history_accordion.children = [text2img_list, img2img_list, controlnet_list, inpainting_list, upscale_list]

history_accordion.set_title(0, "Text-to-Image History 🔮✍")
history_accordion.set_title(1, "Image-to-Image History 🔮🎨")
history_accordion.set_title(2, "ControlNet History 🔮🔧")
history_accordion.set_title(3, "Inpainting History 🔮🖌️")
history_accordion.set_title(4, "Image Upscaler History 🔮✨")

history_display_vbox = widgets.VBox([history_accordion, history_image_display_first], continuous_update = True)
