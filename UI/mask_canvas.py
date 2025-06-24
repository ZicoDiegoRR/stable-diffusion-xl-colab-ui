from IPython.display import display, clear_output
from ipycanvas import MultiCanvas
import ipywidgets as widgets
from io import BytesIO
from PIL import Image

class MaskCanvas:
    # BRUSH
    #______________________________________________________________________________________________________
    def brush(self, x, y):
        if not self.delete_brush.value:
            self.foreground.fill_style = "white"
        else:
            self.foreground.fill_style = "black"
        self.foreground.fill_circle(x, y, self.brush_size.value)
    
    def foreground_on_down(self, x, y):
        self.draw = True
        self.brush(x, y)

    def foreground_on_move(self, x, y):
        if self.draw:
            self.brush(x, y)

    def foreground_on_release(self, x, y):
        self.draw = False

    def brush_preview_move(self, x, y):
        self.brush_preview.clear()
        self.brush_preview.fill_style = "black"
        self.brush_preview.fill_circle(x, y, self.brush_size.value)
    #______________________________________________________________________________________________________

    # INITIALIZATION
    #______________________________________________________________________________________________________
    def buffer(self, image):
        buffer = BytesIO()
        with buffer:
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            image_io = buffer.read()

        return image_io

    def wrap_settings(self):
        return self.canvas_settings

    def canvas_setup(self, image):
        img_widget = widgets.Image(
            value=self.buffer(image)
        )
        self.background.draw_image(img_widget)
        self.background.filter = "brightness(78%) grayscale(5%)"

        blank_widget = widgets.Image(
            value=self.buffer(Image.new("RGB", (self.canvas_width, self.canvas_height), color=(0, 0, 0)))
        )
        self.foreground.draw_image(blank_widget)
        self.foreground.global_alpha = 0.75

        self.brush_preview.global_alpha = 0.35
        
    def create(self, img):
        image = img
        self.width, self.height = img.size
        self.draw = False
        
        self.width_to_max = 1
        if self.width != 512:
            width_to_max = self.width / 512
            width_factor = width_to_max**(-1)
            image.resize((512, int(self.height*width_factor)))

        self.canvas_width, self.canvas_height = image.size
        self.canvas.width = self.canvas_width
        self.canvas.height = self.canvas_height

        self.background = self.canvas[0]
        self.foreground = self.canvas[1]
        self.brush_preview = self.canvas[2]

        self.canvas_setup(image)

        self.foreground.on_mouse_down(self.foreground_on_down)
        self.foreground.on_mouse_move(self.foreground_on_move)
        self.foreground.on_mouse_up(self.foreground_on_release)

        self.brush_preview.on_mouse_move(self.brush_preview_move)
    
    def __init__(self):
        self.canvas = MultiCanvas(3, width=512, height=512)

        self.brush_label = widgets.Label(value="Brush size:")
        self.brush_size = widgets.IntSlider(min=1, max=25, value=3)
        
        self.undo_button = widgets.Button(description="↻")
        self.delete_brush = widgets.Checkbox(description="✂️")
        self.clear_canvas = widgets.Button(description="❌")
        
        self.submit_button = widgets.Button(description="✅")
        self.back_button = widgets.Button(description="🔙")

        self.canvas_settings = widgets.HBox([
            widgets.VBox([
                self.brush_label,
                self.brush_size,
                widgets.HBox([
                    self.undo_button, self.delete_brush, self.clear_canvas,
                ]),
                self.canvas,
                widgets.HBox([
                    self.submit_button, self.clear_canvas,
                ]),
            ]),
        ], justify_content="center")
