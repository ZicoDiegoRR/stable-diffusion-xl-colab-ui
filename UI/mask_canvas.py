from IPython.display import display, clear_output
from ipycanvas import MultiCanvas
import ipywidgets as widgets
from io import BytesIO
from PIL import Image
import numpy as np
import time
import os

class MaskCanvas:
    # BRUSH
    #______________________________________________________________________________________________________
    # Create circles
    def brush(self, x, y):
        self.foreground.global_alpha = 1
        self.foreground.fill_style = "white"
        self.foreground.fill_circle(x, y, self.brush_size.value)
    
    # Enable drawing when clicking the left mouse button
    def foreground_on_down(self, x, y):
        self.draw = True
        self.foreground.sync_image_data = True
        self.brush(x, y)
        self.collected_points.append([x, y])

    # Draw the circles or show the brush preview
    def foreground_on_move(self, x, y):
        if self.draw:
            self.brush_preview.clear()
            self.brush(x, y)
            self.collected_points.append([x, y])
        else:
            self.brush_preview_move(x, y)

    # Disable drawing after releasing the left mouse button
    def foreground_on_release(self, x, y):
        self.draw = False
        self.collected_brushes.append(self.collected_points)
        self.collected_points = []
        time.sleep(1) # Giving the canvas time to update
        self.foreground.sync_image_data = False

    # Create a circle for preview
    def brush_preview_move(self, x, y):
        self.brush_preview.clear()
        self.brush_preview.fill_style = "black"
        self.brush_preview.fill_circle(x, y, self.brush_size.value)

    # Undo changes
    def undo(self):
        self.foreground.sync_image_data = True
        if len(self.collected_brushes) > 0:
            self.collected_brushes.pop(-1)
            self.foreground.clear()
            for brush in self.collected_brushes:
                for point in brush:
                    self.brush(point[0], point[1])
        self.foreground.sync_image_data = False

    #______________________________________________________________________________________________________

    # INITIALIZATION
    #______________________________________________________________________________________________________
    # Return a black image
    def black_image(self, width, height):
        return Image.new("RGB", (width, height), color=(0, 0, 0, 255))

    # Collect every widget into a single VBox
    def wrap_settings(self):
        return self.mask_ui

    # Return the back button
    def get_back_button(self):
        return self.back_button

    # Return the submit button
    def get_submit_button(self):
        return self.submit_button

    # Disable buttons
    def disable_button(self):
        for button in [self.submit_button, self.back_button, self.reload]:
            button.disabled = True

    # Enable buttons
    def enable_button(self):
        for button in [self.submit_button, self.back_button, self.reload]:
            button.disabled = False

    # Convert PIL images into bytes
    def buffer(self, image):
        buffer = BytesIO()
        with buffer:
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            image_io = buffer.read()

        return image_io

    def save_mask(self):
        self.disable_button()
        self.foreground.sync_image_data = True
        time.sleep(5) # Giving the canvas time to update

        os.makedirs("/content/mask", exist_ok=True)
        arr_data = self.foreground.get_image_data()

        alpha = arr_data[..., 3:]/255.0
        rgb = (arr_data[..., :3] * alpha).astype(np.uint8)

        converted_image = Image.fromarray(rgb)
        converted_image.resize((self.width, self.height)).save(f"/content/mask/temp.png")
        self.foreground.sync_image_data = False

        self.preview.value = self.buffer(converted_image)
        self.mask_ui.children = [self.canvas_settings, self.preview_section]

        self.enable_button()

    # Stylize the canvas
    def canvas_setup(self, image):
        img_widget = widgets.Image(
            value=self.buffer(image)
        )
        self.background.draw_image(img_widget)

        blank_widget = widgets.Image(
            value=self.buffer(self.black_image(self.canvas_width, self.canvas_height))
        )
        self.blocking.global_alpha = 0.75
        self.blocking.draw_image(blank_widget)

        self.brush_preview.global_alpha = 0.35
        
    # Create a new instance
    def create(self, img):
        self.image = img
        self.width, self.height = img.size
        self.draw = False
        
        self.width_to_max = 1
        if self.width != 256:
            self.width_to_max = self.width / 256
            width_factor = self.width_to_max**(-1)
            self.image = self.image.resize((256, int(self.height*width_factor)))

        self.canvas_width, self.canvas_height = self.image.size
        self.canvas.width = self.canvas_width
        self.canvas.height = self.canvas_height

        self.background = self.canvas[0]
        self.blocking = self.canvas[1]
        self.foreground = self.canvas[2]
        self.brush_preview = self.canvas[3]

        self.canvas_setup(self.image)

        self.brush_preview.on_mouse_down(self.foreground_on_down)
        self.brush_preview.on_mouse_move(self.foreground_on_move)
        self.brush_preview.on_mouse_up(self.foreground_on_release)

    # Clearing the canvas
    def clear(self):
        self.foreground.sync_image_data = True
        self.foreground.clear()
        self.collected_points = []
        self.collected_brushes = []
        self.foreground.sync_image_data = False

    # Reload the canvas (deprecated)
    def reload_canvas(self):
        self.clear()
        self.canvas.clear()
        self.canvas.flush()

        self.create(self.image)
    
    # Initialize everything
    def __init__(self):
        self.collected_points = []
        self.collected_brushes = []
        self.canvas = MultiCanvas(4, width=256, height=256)

        self.preview_label = widgets.Label(value="Is this correct? Try to click the save button again if it's incorrect.")
        self.preview = widgets.Image()
        self.preview_section = widgets.VBox([self.preview_label, widgets.Label(value=" "), self.preview])

        self.brush_label = widgets.Label(value="Brush size:")
        self.brush_size = widgets.IntSlider(min=1, max=25, value=3)
        
        top_button_layout = widgets.Layout(width="40px")
        self.undo_button = widgets.Button(description="↻", layout=top_button_layout)
        self.clear_canvas = widgets.Button(description="❌", layout=top_button_layout)
        self.undo_button.on_click(lambda b: self.undo())
        self.clear_canvas.on_click(lambda b: self.clear())
        
        self.submit_button = widgets.Button(description="✅")
        self.back_button = widgets.Button(description="🔙")
        self.submit_button.on_click(lambda b: self.save_mask())

        self.reload = widgets.Button(description="Reload Canvas")
        self.reload.on_click(lambda b: self.reload_canvas())

        self.canvas_settings = widgets.HBox([
            widgets.VBox([
                widgets.HTML(value="<hr>"),
                self.brush_label,
                widgets.HBox([
                    self.brush_size, self.undo_button, self.clear_canvas,
                ]),
                self.canvas,
                widgets.HBox([
                    self.submit_button, self.back_button, self.reload
                ]),
                widgets.HTML(value="<hr>"),
            ]),
        ], justify_content="center")

        self.mask_ui = widgets.HBox([self.canvas_settings])
