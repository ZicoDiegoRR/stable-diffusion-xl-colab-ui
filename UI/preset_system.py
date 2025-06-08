from StableDiffusionXLColabUI.utils import save_file_converter
from StableDiffusionXLColabUI.UI import all_widgets
from IPython.display import display, HTML
import ipywidgets as widgets
import threading
import json
import time
import os

class PresetSystem:
    # Saving a preset
    def save_param(self, path, param):
        with open(path, "w") as file:
            json.dump(param, file)

    # Loading a preset from a JSON file
    def load_param(self, path):
        try:
            with open(path, "r") as file:
                params = json.load(file)
            return params
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    # Showing an error message 
    def show_message(self, widget, msg, type, sec=2):
        widget.clear_output()
        with widget:
            if type == "error":
                display(HTML(f"<span style='color: red;'>Error:</span> {msg}"))
            elif type == "warn":
                display(HTML(f"<span style='color: orange;'>Warning:</span> {msg}"))
            elif type == "success":
                display(HTML(f"<span style='color: lime;'>Success:</span> {msg}"))
        if type != "warn":
            threading.Timer(sec, widget.clear_output).start()

    # Validating and converting old preset to the new one
    def list_or_dict(self, cfg, path):
        if isinstance(cfg, list):
            new_cfg = save_file_converter.old_to_new(cfg)
            self.save_param(path, new_cfg)
            return new_cfg
        elif isinstance(cfg, dict):
            return cfg
            
    # Wrapping every widget into a vbox
    def wrap_settings(self):
        return self.preset_tab_vbox

    # Resetting the dropdown options
    def reset_options(self):
        self.load_preset_selection_dropdown.options = self.list_all_saved_preset()
        self.rename_preset_selection_dropdown.options = self.list_all_saved_preset()
        self.delete_preset_selection_dropdown.options = self.list_all_saved_preset()

    # Final phase of saving preset
    def save_warning_evaluate(self, result, name, text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
        if result == "override":
            save_params = all_widgets.import_values(text2img, img2img, controlnet, inpaint, ip, lora, embeddings)
            self.save_param(f"{self.base_path}/Saved Parameters/{name}.json", save_params)
            self.reset_options()
            self.show_message(self.save_output, f"Saved {name}.json in {self.base_path} folder.", "success")

        self.save_preset_display.children = [
            self.save_output, 
            self.save_preset_name_widget, 
            self.save_preset_button,
            widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")
        ]
            
        self.save_preset_button._click_handlers.callbacks.clear()
        self.save_preset_button.on_click(lambda b: self.save_preset_on_click(self.save_preset_name_widget.value, text2img, img2img, controlnet, inpaint, ip, lora, embeddings))

    # First phase of saving preset if another file with the same name exists
    def save_warning_if_preset_exists(self, name, text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
        self.save_warning_back_button._click_handlers.callbacks.clear()
        self.save_preset_display.children = [
            self.save_output, 
            self.save_preset_name_widget, 
            widgets.HBox(
                [
                    self.save_preset_button, 
                    self.save_warning_back_button
                ]
            ), 
            widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")
        ]

        self.show_message(self.save_output, f"{name}.json already exists. Saving the current parameters with the same name will overwrite the original saved parameters. Do you wish to continue?", "warn")
        self.save_warning_back_button.on_click(lambda b: self.save_warning_evaluate("back", name, text2img, img2img, controlnet, inpaint, ip, lora, embeddings))

        self.save_preset_button._click_handlers.callbacks.clear()
        self.save_preset_button.on_click(lambda b: self.save_warning_evaluate("override", name, text2img, img2img, controlnet, inpaint, ip, lora, embeddings))

    # Saving and/or evaluating input
    def save_preset_on_click(self, name, text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
        if name and name not in self.list_all_saved_preset():
            save_params = all_widgets.import_values(text2img, img2img, controlnet, inpaint, ip, lora, embeddings)
            self.save_param(f"{self.base_path}/Saved Parameters/{name}.json", save_params)
            self.show_message(self.save_output, f"Saved {name}.json in {self.base_path} folder.", "success")
            self.reset_options()
        elif name in self.list_all_saved_preset():
            self.save_warning_if_preset_exists(name, text2img, img2img, controlnet, inpaint, ip, lora, embeddings)
        else:
            self.show_message(self.save_output, "Name cannot be empty!", "error")

    # Getting all presets
    def list_all_saved_preset(self):
        list_of_saved_parameters = [word.replace(".json" , "") for word in os.listdir(f"{self.base_path}/Saved Parameters/") if os.path.isfile(os.path.join(f"{self.base_path}/Saved Parameters/", word)) and word.endswith(".json")]
        return list_of_saved_parameters

    # Load a preset into your parameters
    def load_preset_on_click(self, name, text2img, img2img, controlnet, inpaint, ip, lora, embeddings):
        param_path = os.path.join(f"{self.base_path}/Saved Parameters/", f"{name}.json")
        
        preset_raw_cfg = self.load_param(param_path)
        preset_cfg = self.list_or_dict(preset_raw_cfg, param_path)
        
        every_widgets = all_widgets.import_widgets(text2img, img2img, controlnet, inpaint, ip, lora, embeddings)
        for key, items in every_widgets.items():
            for i in range(len(items)):
                items[i].value = preset_cfg[key][i]
                
        lora.construct(preset_cfg["lora"])
        embeddings.construct(preset_cfg["embeddings"])
        self.show_message(self.load_output, f"Loaded {name}.json.", "success")

    # Final phase of renaming preset if another file with the same name exists
    def rename_preset_evaluate(self, result, old, new):
        if result == "overwrite":
            os.rename(os.path.join(f"{self.base_path}/Saved Parameters/", f"{old}.json"), os.path.join(f"{self.base_path}/Saved Parameters/", f"{new}.json"))
            self.reset_options()
            self.show_message(self.rename_output, f"Renamed {old}.json to {new}.json.", "success")

        self.rename_preset_display.children = [
            self.rename_output,
            widgets.HBox([
                widgets.VBox([self.rename_preset_selection_dropdown_label, self.rename_preset_selection_dropdown]), 
                widgets.VBox([self.rename_preset_widget_label, self.rename_preset_widget])
            ]),
            self.rename_preset_button, 
            widgets.HTML(value="Clicking the button will rename your selected saved preset.")
        ]

        self.rename_preset_button._click_handlers.callbacks.clear()
        self.rename_preset_button.on_click(lambda b: self.rename_preset_on_click(self.rename_preset_selection_dropdown.value, self.rename_preset_widget.value))

    # Renaming and/or evaluating input
    def rename_preset_on_click(self, old, new):
        if new not in self.list_all_saved_preset() and new and new != old:
            os.rename(os.path.join(f"{self.base_path}/Saved Parameters/", f"{old}.json"), os.path.join(f"{self.base_path}/Saved Parameters/", f"{new}.json"))
            self.reset_options()
            self.show_message(self.rename_output, f"Renamed {old}.json to {new}.json.", "success")
        elif new in self.list_all_saved_preset() and new and new != old:
            self.rename_back_button._click_handlers.callbacks.clear()
            self.show_message(self.rename_output, f"{new}.json already exists. Renaming the current parameters with the same name will overwrite the original saved parameters. Do you wish to continue?", "warn")
            self.rename_preset_button._click_handlers.callbacks.clear()
            
            self.rename_preset_display.children = [
                self.rename_output,
                widgets.HBox([
                    widgets.VBox([self.rename_preset_selection_dropdown_label, self.rename_preset_selection_dropdown]), 
                    widgets.VBox([self.rename_preset_widget_label, self.rename_preset_widget])
                ]),
                widgets.HBox([self.rename_preset_button, self.rename_back_button]), 
                widgets.HTML(value="Clicking the button will rename your selected saved preset.")
            ]

            self.rename_back_button.on_click(lambda b: self.rename_preset_evaluate("back", old, new))
            self.rename_preset_button.on_click(lambda b: self.rename_preset_evaluate("overwrite", old, new))
        elif new == old or not new:
            if new:
                rename_error_message = "New name cannot be the same as the old name!"
            else:
                rename_error_message = "New name cannot be empty!"
            self.show_message(self.rename_output, rename_error_message, "error")

    # Final phase of deleting a preset
    def delete_preset_evaluate(self, result, name):
        if result == "delete":
            os.remove(os.path.join(f"{self.base_path}/Saved Parameters/", f"{name}.json"))
            self.reset_options()
            self.show_message(self.delete_output, f"Deleted {name}.json.", "success")

        self.delete_preset_display.children = [
            self.delete_output, 
            self.delete_preset_selection_dropdown_label, 
            self.delete_preset_selection_dropdown, 
            self.delete_preset_button, 
            widgets.HTML(value="Clicking the button will delete your selected saved preset from Google Drive.")]

        self.delete_preset_button._click_handlers.callbacks.clear()
        self.delete_preset_button.on_click(lambda b: self.delete_preset_on_click(self.delete_preset_selection_dropdown.value))

    # Confirming the user's decision
    def delete_preset_on_click(self, name):
        self.delete_back_button._click_handlers.callbacks.clear()

        self.show_message(self.delete_output, f"Do you wish to delete {name}.json from your saved presets?", "warn")
        self.delete_preset_button._click_handlers.callbacks.clear()

        self.delete_preset_display.children = [
            self.delete_output, 
            self.delete_preset_selection_dropdown_label, 
            self.delete_preset_selection_dropdown, 
            widgets.HBox([self.delete_preset_button, self.delete_back_button]), 
            widgets.HTML(value="Clicking the button will delete your selected saved preset from Google Drive.")]

        self.delete_back_button.on_click(lambda b: self.delete_preset_evaluate("back", name))
        self.delete_preset_button.on_click(lambda b: self.delete_preset_evaluate("delete", name))

    # Initiating widgets creation
    def __init__(self, text2img, img2img, controlnet, inpaint, ip, lora, embeddings, base_path):
        self.base_path = base_path
        self.rename_output = widgets.Output()
        self.save_output = widgets.Output()
        self.load_output = widgets.Output()
        self.delete_output = widgets.Output()
        
        self.save_warning_back_button = widgets.Button(description="Back")
        self.save_preset_button = widgets.Button(description="Save current parameters")
        self.save_preset_name_widget = widgets.Text(description="Name", placeholder="Preset name", value="")
        self.save_preset_display = widgets.VBox([self.save_output, self.save_preset_name_widget, self.save_preset_button, widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")])

        self.load_preset_button = widgets.Button(description="Load this preset")
        self.load_preset_selection_dropdown = widgets.Dropdown(description="Select your saved preset")
        self.load_preset_display = widgets.VBox([self.load_output, self.load_preset_selection_dropdown, self.load_preset_button, widgets.HTML(value="Clicking the load button will override the current parameters.")])

        self.rename_preset_button = widgets.Button(description="Rename this preset")
        self.rename_back_button = widgets.Button(description="Back")
        self.rename_preset_selection_dropdown_label = widgets.Label(value="Select your saved preset:")
        self.rename_preset_selection_dropdown = widgets.Dropdown()
        self.rename_preset_widget_label = widgets.Label(value="Input your new name:")
        self.rename_preset_widget = widgets.Text(placeholder="New name")
        self.rename_preset_display = widgets.VBox([self.rename_output, widgets.HBox([widgets.VBox([self.rename_preset_selection_dropdown_label, self.rename_preset_selection_dropdown]), widgets.VBox([self.rename_preset_widget_label, self.rename_preset_widget])]), self.rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")])

        self.delete_preset_button = widgets.Button(description="Delete this preset", button_style="danger")
        self.delete_back_button = widgets.Button(description="Back")
        self.delete_preset_selection_dropdown_label = widgets.Label(value="Select your saved preset:")
        self.delete_preset_selection_dropdown = widgets.Dropdown()
        self.delete_preset_display = widgets.VBox([self.delete_output, self.delete_preset_selection_dropdown_label, self.delete_preset_selection_dropdown, self.delete_preset_button, widgets.HTML(value="Clicking the button will delete your selected saved preset from Google Drive.")])

        self.reset_options()
        
        self.preset_tab = widgets.Tab()
        self.preset_tab.layout = widgets.Layout(width="99%")
        self.preset_tab.children = [self.save_preset_display, self.load_preset_display, self.rename_preset_display, self.delete_preset_display]
        self.preset_tab.set_title(0, "Save Preset")
        self.preset_tab.set_title(1, "Load Preset")
        self.preset_tab.set_title(2, "Rename Preset")
        self.preset_tab.set_title(3, "Delete Preset")

        self.save_preset_button.on_click(lambda b: self.save_preset_on_click(self.save_preset_name_widget.value, text2img, img2img, controlnet, inpaint, ip, lora, embeddings))
        self.load_preset_button.on_click(lambda b: self.load_preset_on_click(self.load_preset_selection_dropdown.value, text2img, img2img, controlnet, inpaint, ip, lora, embeddings))
        self.rename_preset_button.on_click(lambda b: self.rename_preset_on_click(self.rename_preset_selection_dropdown.value, self.rename_preset_widget.value))
        self.delete_preset_button.on_click(lambda b: self.delete_preset_on_click(self.delete_preset_selection_dropdown.value))

        self.preset_tab_vbox = widgets.VBox([self.preset_tab], layout=widgets.Layout(width="50%", align_items="flex-end"))
