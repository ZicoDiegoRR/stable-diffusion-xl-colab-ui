from IPython.display import display, clear_output
from ipycanvas import MultiCanvas
import ipywidgets as widgets
from PIL import Image

class MaskCanvas:
    def wrap_settings(self):
        return self.canvas_settings

    def canvas_setup(self, image):
        img_widget = widgets.Image(
            value=open(image, "rb").read()
        )
        self.canvas[0].draw_image(img_widget)

        blank_widget = widgets.Image(
            value=open(Image.new(
                "RGB", 
                (256, 256), 
                color=(0, 0, 0))
            ).read()
        )
        
        
    def create(self, img):
        image = img
        width, height = img.size
        if width != 512:
            width_to_max = width / 512
            width_factor = width_to_max**(-1)
            image.resize((512, int(height*width_factor)))

        canvas_width, canvas_height = image.size
        self.canvas.width = canvas_width
        self.canvas.height = canvas_height

    def __init__(self):
        self.canvas = MultiCanvas(3, width=512, height=512)

        self.undo_button = widgets.Button(description="↻")
        self.brush_size = widgets.IntSlider(min=1, max=25, value=3, description="Brush size")
        self.delete_brush = widgets.Checkbox(description="🧼")
        self.clear_canvas = widgets.Button(description="❌")
        self.submit_button = widgets.Button(description="✅")

        self.canvas_settings = widgets.HBox([
            widgets.VBox([
                widgets.HBox([
                    self.undo_button, self.delete_brush,
                ]),
                self.canvas,
                widgets.HBox([
                    self.submit_button, self.clear_canvas,
                ]),
            ]),
            self.brush_size,
        ], justify_content="center")
        
