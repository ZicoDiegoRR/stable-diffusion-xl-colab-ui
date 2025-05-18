from StableDiffusionXLColabUI.UI import all_widgets
import ipywidgets as widgets
import json
import time
import os

class PresetSystem:
    def save_param(self, path, param):
        with open(path, "w") as file:
            json.dump(param, file)
            
    def save_warning_evaluate(self, result, name):
        if result == "override":
            save_params = all_widgets.import_values()
            self.save_param(f"{base_path}/Saved Parameters/{name}.json", save_params)
            self.load_preset_selection_dropdown.options = self.list_all_saved_preset()
            self.rename_preset_selection_dropdown.options = self.list_all_saved_preset()
            self.delete_preset_selection_dropdown.options = self.list_all_saved_preset()

            self.save_preset_display.children = [
                self.save_preset_name_widget, self.save_preset_button, widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.") 
            ] if result != "override" else [
                widgets.HTML(value=f"Succesfully saved the current parameters as {name}.json in '{base_path}/Saved Parameters' folder."), self.save_preset_name_widget, self.save_preset_button, widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")
            ]
            self.save_preset_button._click_handlers.callbacks.clear()
            self.save_preset_button.on_click(lambda b: self.save_preset_on_click(self.save_preset_name_widget.value))

    def save_warning_if_preset_exists(self, name):
        self.save_warning_back_button._click_handlers.callbacks.clear()

        self.save_preset_display.children = [widgets.HTML(value=f"<span style='color: orange;'>Warning:</span> {name}.json already exists. Saving the current parameters with the same name will overwrite the original saved parameters. Do you wish to continue?"), self.save_preset_name_widget, widgets.HBox([self.save_preset_button, self.save_warning_back_button])]
        self.save_warning_back_button.on_click(lambda b: self.save_warning_evaluate("back", name))

        self.save_preset_button._click_handlers.callbacks.clear()
        self.save_preset_button.on_click(lambda b: self.save_warning_evaluate("override", name))

    def save_preset_on_click(name):
        if name and name not in self.list_all_saved_preset():
            save_params = all_widgets.import_values()
            self.save_param(f"{base_path}/Saved Parameters/{name}.json", save_params)
            self.save_preset_display.children = [widgets.HTML(value=f"Succesfully saved the current parameters as {name}.json in {base_path}/Saved Parameters folder."), self.save_preset_name_widget, self.save_preset_button, widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")]
            self.load_preset_selection_dropdown.options = self.list_all_saved_preset()
            self.rename_preset_selection_dropdown.options = self.list_all_saved_preset()
            self.delete_preset_selection_dropdown.options = self.list_all_saved_preset()
        elif name in self.list_all_saved_preset():
            self.save_warning_if_preset_exists(name)
        else:
            self.save_preset_display.children = [widgets.HTML(value="<span style='color: red;'>Error:</span> Name cannot be empty!"), self.save_preset_name_widget, self.save_preset_button, widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")]
            time.sleep(1.5)
            self.save_preset_display.children = [self.save_preset_name_widget, self.save_preset_button, widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")]

    def list_all_saved_preset(self):
        list_of_saved_parameters = [word.replace(".json" , "") for word in os.listdir(f"{base_path}/Saved Parameters/") if os.path.isfile(os.path.join(f"{base_path}/Saved Parameters/", word)) and word.endswith(".json")]
        return list_of_saved_parameters

    def load_preset_on_click(self, name):
        preset_cfg = load_param(os.path.join(f"{base_path}/Saved Parameters/", f"{name}.json"))
  every_widgets = all_widgets()
  for i in range(len(every_widgets.children)):
    every_widgets.children[i].value = preset_cfg[i]
  lora_reader_upon_starting()
  ti_reader_upon_starting()

def rename_preset_evaluate(result, old, new):
  if result == "overwrite":
    os.rename(os.path.join(f"{base_path}/Saved Parameters/", f"{old}.json"), os.path.join(f"{base_path}/Saved Parameters/", f"{new}.json"))
    load_preset_selection_dropdown.options = list_all_saved_preset()
    rename_preset_selection_dropdown.options = list_all_saved_preset()
    delete_preset_selection_dropdown.options = list_all_saved_preset()
    rename_preset_display.children = [widgets.HTML(value=f"Succesfully renamed {old}.json to {new}.json."), widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")]
  else:
    rename_preset_display.children = [widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")]

  rename_preset_button._click_handlers.callbacks.clear()
  rename_preset_button.on_click(lambda b: rename_preset_on_click(rename_preset_selection_dropdown.value, rename_preset_widget.value))

def rename_preset_on_click(old, new):
  if new not in list_all_saved_preset() and new and new != old:
    os.rename(os.path.join(f"{base_path}/Saved Parameters/", f"{old}.json"), os.path.join(f"{base_path}/Saved Parameters/", f"{new}.json"))
    load_preset_selection_dropdown.options = list_all_saved_preset()
    rename_preset_selection_dropdown.options = list_all_saved_preset()
    delete_preset_selection_dropdown.options = list_all_saved_preset()
    rename_preset_display.children = [widgets.HTML(value=f"Succesfully renamed {old}.json to {new}.json."), widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")]
  elif new in list_all_saved_preset() and new and new != old:
    rename_back_button = widgets.Button(description="Back")
    rename_back_button._click_handlers.callbacks.clear()

    rename_preset_display.children = [widgets.HTML(value=f"<span style='color: orange;'>Warning:</span> {new}.json already exists. Rename the current parameters with the same name will overwrite the original saved parameters. Do you wish to continue?"), widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), widgets.HBox([rename_preset_button, rename_back_button]), widgets.HTML(value="Clicking the button will rename your selected saved preset.")]
    rename_preset_button._click_handlers.callbacks.clear()

    rename_back_button.on_click(lambda b: rename_preset_evaluate("back", old, new))
    rename_preset_button.on_click(lambda b: rename_preset_evaluate("overwrite", old, new))
  elif new == old:
    rename_preset_display.children = [widgets.HTML(value=f"<span style='color: red;'>Error:</span> New name cannot be the same as the old name!"), widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")]
    time.sleep(1.5)
    rename_preset_display.children = [widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]),rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")]
  elif not new:
    rename_preset_display.children = [widgets.HTML(value=f"<span style='color: red;'>Error:</span> New name cannot be empty!"), widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")]
    time.sleep(1.5)
    rename_preset_display.children = [widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")]

def delete_preset_evaluate(result, name):
  if result == "delete":
    os.remove(os.path.join(f"{base_path}/Saved Parameters/", f"{name}.json"))
    load_preset_selection_dropdown.options = list_all_saved_preset()
    rename_preset_selection_dropdown.options = list_all_saved_preset()
    delete_preset_selection_dropdown.options = list_all_saved_preset()
    delete_preset_display.children = [widgets.HTML(value=f"Successfully deleted {name}.json."), delete_preset_selection_dropdown_label, delete_preset_selection_dropdown, delete_preset_button, widgets.HTML(value="Clicking the button will delete your selected saved preset from Google Drive.")]
  else:
    delete_preset_display.children = [delete_preset_selection_dropdown_label, delete_preset_selection_dropdown, delete_preset_button, widgets.HTML(value="Clicking the button will delete your selected saved preset from Google Drive.")]

  delete_preset_button._click_handlers.callbacks.clear()
  delete_preset_button.on_click(lambda b: delete_preset_on_click(delete_preset_selection_dropdown.value))

def delete_preset_on_click(name):
  delete_back_button = widgets.Button(description="Back")
  delete_back_button._click_handlers.callbacks.clear()

  delete_preset_display.children = [widgets.HTML(value=f"<span style='color: orange;'>Warning:</span> Do you wish to delete {name}.json from your saved presets?"), delete_preset_selection_dropdown_label, delete_preset_selection_dropdown, widgets.HBox([delete_preset_button, delete_back_button]), widgets.HTML(value="Clicking the button will delete your selected saved preset from Google Drive.")]
  delete_preset_button._click_handlers.callbacks.clear()

  delete_back_button.on_click(lambda b: delete_preset_evaluate("back", name))
  delete_preset_button.on_click(lambda b: delete_preset_evaluate("delete", name))

save_warning_back_button = widgets.Button(description="Back")
save_preset_button = widgets.Button(description="Save current parameters")
save_preset_name_widget = widgets.Text(description="Name", placeholder="Preset name", value="")
save_preset_display = widgets.VBox([save_preset_name_widget, save_preset_button, widgets.HTML(value="Clicking the save button will save the current parameters you're using as a new preset for later use.")])

load_preset_button = widgets.Button(description="Load this preset")
load_preset_selection_dropdown = widgets.Dropdown(description="Select your saved preset")
load_preset_selection_dropdown.options = list_all_saved_preset()
load_preset_display = widgets.VBox([load_preset_selection_dropdown, load_preset_button, widgets.HTML(value="Clicking the load button will override the current parameters.")])

rename_preset_button = widgets.Button(description="Rename this preset")
rename_preset_selection_dropdown_label = widgets.Label(value="Select your saved preset:")
rename_preset_selection_dropdown = widgets.Dropdown()
rename_preset_selection_dropdown.options = list_all_saved_preset()
rename_preset_widget_label = widgets.Label(value="Input your new name:")
rename_preset_widget = widgets.Text(placeholder="New name")
rename_preset_display = widgets.VBox([widgets.HBox([widgets.VBox([rename_preset_selection_dropdown_label, rename_preset_selection_dropdown]), widgets.VBox([rename_preset_widget_label, rename_preset_widget])]), rename_preset_button, widgets.HTML(value="Clicking the button will rename your selected saved preset.")])

delete_preset_button = widgets.Button(description="Delete this preset", button_style="danger")
delete_preset_selection_dropdown_label = widgets.Label(value="Select your saved preset:")
delete_preset_selection_dropdown = widgets.Dropdown()
delete_preset_selection_dropdown.options = list_all_saved_preset()
delete_preset_display = widgets.VBox([delete_preset_selection_dropdown_label, delete_preset_selection_dropdown, delete_preset_button, widgets.HTML(value="Clicking the button will delete your selected saved preset from Google Drive.")])

preset_tab = widgets.Tab()
preset_tab.layout = widgets.Layout(width="99%")
preset_tab.children = [save_preset_display, load_preset_display, rename_preset_display, delete_preset_display]
preset_tab.set_title(0, "Save Preset")
preset_tab.set_title(1, "Load Preset")
preset_tab.set_title(2, "Rename Preset")
preset_tab.set_title(3, "Delete Preset")

save_preset_button.on_click(lambda b: save_preset_on_click(save_preset_name_widget.value))
load_preset_button.on_click(lambda b: load_preset_on_click(load_preset_selection_dropdown.value))
rename_preset_button.on_click(lambda b: rename_preset_on_click(rename_preset_selection_dropdown.value, rename_preset_widget.value))
delete_preset_button.on_click(lambda b: delete_preset_on_click(delete_preset_selection_dropdown.value))

preset_tab_vbox = widgets.VBox([preset_tab], layout=widgets.Layout(width="50%", align_items="flex-end"))
