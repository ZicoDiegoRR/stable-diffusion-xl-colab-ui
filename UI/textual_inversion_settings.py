import ipywidgets as widgets
import json
import re
import os

def load_param(filename):
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        return {}

class TextualInversionLoader:
    def collect_values(self): # Function to collect values in this class
        self.ti_urls_widget.value, self.ti_tokens_widget.value = self.read()
        return [self.ti_urls_widget.value, self.ti_tokens_widget.value]

    def wrap_settings(self):
        return self.ti_settings

    def return_widgets(self):
        return [
            self.ti_urls_widget,
            self.ti_tokens_widget
        ]

    def refresh_model(self):
        saved_models = load_param(f"{self.base_path}/Saved Parameters/URL/urls.json").get("Embeddings")
        saved_hf_models = saved_models["hugging_face"] if saved_models and "hugging_face" in saved_models else []
        if not saved_models:
            model_options = []
        else:
            model_options = list(saved_models["keyname_to_url"].keys())
        return model_options + saved_hf_models
        
    def sanitize(self, cfg, token): # Function to filter out empty strings and invalid path
        if cfg:
            split_cfg = re.split(r"\s*,\s*", cfg)
            split_token = re.split(r"\s*,\s*", token)
            sanitized_cfg = ""
            sanitized_token = ""
            for item, tag in zip(split_cfg, split_token):
                sanitized_cfg += item + ","
                sanitized_token += tag + ","
            return sanitized_cfg.rstrip(","), sanitized_token.rstrip(",")
        else:
            return "", ""
        
    def ti_click(self, link, token, construct=False):  # Function to add widgets after clicking the plus button
        ti_url_input = widgets.Combobox(value=link, options=self.refresh_model(), placeholder="Input the link here", description="Weight file", ensure_option=False)
        ti_tokens_input = widgets.Text(value=token, placeholder="Activation tag", description="Token")
        ti_remove_button = widgets.Button(description="X", button_style='danger', layout=widgets.Layout(width='30px', height='30px'))

        if construct and not self.ti_construct_bool:
            self.ti_construct_bool = True
            self.ti_nested_vbox.children = []
        
        self.ti_nested_vbox.children += (ti_url_input, ti_tokens_input, ti_remove_button,)
        ti_remove_button.on_click(lambda b: self.ti_remover(
            list(self.ti_nested_vbox.children).index(ti_remove_button) - 2,
            list(self.ti_nested_vbox.children).index(ti_remove_button) - 1,
            list(self.ti_nested_vbox.children).index(ti_remove_button)
        ))
        self.ti_settings.children = [self.ti_tip, self.ti_add, self.ti_nested_vbox]

    def read(self):  # Function to process every value into two strings to be fed into the main logic
        collected_ti_urls = ""
        collected_ti_tokens = ""
        for i in range(len(self.ti_nested_vbox.children)):
            if i % 3 == 0:
              if self.ti_nested_vbox.children[i].value != "":
                collected_ti_urls += (self.ti_nested_vbox.children[i].value + ",")
            elif i % 3 == 1:
              if self.ti_nested_vbox.children[i - 1].value != "":
                collected_ti_tokens += (self.ti_nested_vbox.children[i].value + ",")
        return collected_ti_urls.rstrip(","), collected_ti_tokens.rstrip(",")

    def ti_remover(self, link, token, remove_button):  # Function to remove textual inversion widgets
        ti_nested_list = list(self.ti_nested_vbox.children)
        ti_nested_list.pop(remove_button)
        ti_nested_list.pop(token)
        ti_nested_list.pop(link)
        self.ti_nested_vbox.children = tuple(ti_nested_list)

    def construct(self, cfg):  # Function to add widgets from saved parameters
        ti_links = re.split(r"\s*,\s*", self.ti_urls_widget.value)
        ti_tokens = re.split(r"\s*,\s*", self.ti_tokens_widget.value)
        if len(ti_tokens) < len(ti_links):
            for i in range(len(ti_links) - len(ti_tokens)):
                ti_tokens.append("")
        for i, embeddings in enumerate(ti_links):
            if embeddings:
                self.ti_click(ti_links[i], ti_tokens[i], construct=True)
        self.ti_construct_bool = False

    def __init__(self, cfg, base_path):
        self.base_path = base_path
        
        sanitized_url, sanitized_token = self.sanitize(cfg[0], cfg[1])
        self.ti_urls_widget = widgets.Text(value=sanitized_url)
        self.ti_tokens_widget = widgets.Text(value=sanitized_token)

        self.ti_add = widgets.Button(description="+", button_style='success', layout=widgets.Layout(width='30px', height='30px'))
        self.ti_nested_vbox = widgets.VBox()
        self.ti_tip = widgets.HTML(value="Due to the architecture, you must pass the activation tag in the Token widget. Leaving it blank will skip the embeddings from being loaded to avoid any issue.")
        self.ti_settings = widgets.VBox([self.ti_tip, self.ti_add])

        self.ti_construct_bool = False

        self.ti_add.on_click(lambda b: self.ti_click("", ""))
        self.construct(cfg)
