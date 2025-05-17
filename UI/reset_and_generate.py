from StableDiffusionXLColabUI.UI import all_widgets
import ipywidgets as widgets

def param_default():
    default_param = {
        "text2img": ["", "", "", 1024, 1024, 12, 6, 2, "Default (defaulting to the model)", 
                     False, False, False, False, "", "", False,
                    ],
        "img2img": ["", "", "", 1024, 1024, 12, 6, 2, "Default (defaulting to the model)", 
                    False, False, False, False, "", "", False, "", 0.3,
                   ],
        "controlnet": ["", "", "", 1024, 1024, 12, 6, 2, "Default (defaulting to the model)", 
                      False, False, False, False, "", "", False, "", 100, 240, False, 
                      0.7, "", False, 0.7, "", False, 0.7,
                      ],
        "inpaint": ["pre-generated text2image image", "", False, 0.9],
        "ip": ["", 0.8, "None"],
        "lora": ["", ""],
        "embeddings": ["", ""],
    }

class ResetGenerateSettings:
    def reset_evaluate(self, result): # Function to set every parameter into the default value
        if result == "yes":
            cfg_reset = param_default()
            every_widgets = all_widgets.import_widgets()
            for key, items in every_widgets.items():
                for item, reset in zip(items, cfg_reset[key]):
                    item.value = reset
        self.reset_display.children = [self.reset_button]

    def reset_button_click(self): # Function to show a warning when the reset parameter button is clicked
        self.reset_display.children = [widgets.HTML(value="Are you sure you want to reset all parameters to default? You still can revert it back after rerunning this cell. LoRA and embeddings won't be reset."), widgets.HBox([self.reset_yes_button, self.reset_no_button])]
        self.reset_yes_button._click_handlers.callbacks.clear()
        self.reset_no_button._click_handlers.callbacks.clear()

        self.reset_yes_button.on_click(lambda b: self.reset_evaluate("yes"))
        self.reset_no_button.on_click(lambda b: self.reset_evaluate("no"))

    def __init__(self):
        self.submit_button_widget = widgets.Button(disabled=False, button_style='', description="Generate")
        self.dont_spam = widgets.HTML(value="Please <b>don't spam</b> the generate button!")
        self.keep_generating = widgets.HTML(value="You still can generate even though the cell is complete executing.")
        self.submit_display = widgets.VBox([self.submit_button_widget, self.dont_spam, self.keep_generating], layout=widgets.Layout(width="50%"))

        self.reset_yes_button = widgets.Button(description="Yes", button_style='danger')
        self.reset_no_button = widgets.Button(description="No")
        self.reset_button = widgets.Button(description="Reset parameters to default")
        self.reset_display = widgets.VBox([self.reset_button])

        self.reset_button.on_click(lambda b: self.reset_button_click())
