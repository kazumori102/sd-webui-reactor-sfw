import gradio as gr
from scripts.reactor_logger import logger
from scripts.reactor_helpers import get_models, set_Device
from scripts.reactor_globals import DEVICE, DEVICE_LIST

def update_models_list(selected: str):
    choices = get_models()
    return gr.update(value=selected if selected in choices else next(iter(choices), None), choices=choices)

def show(hash_check_block: bool = True):
    # TAB SETTINGS
    with gr.Tab("Settings"):
        models = get_models()
        with gr.Row():
            device = gr.Radio(
                label="Execution Provider",
                choices=DEVICE_LIST,
                value=DEVICE,
                type="value",
                info="Available providers in ONNX Runtime. Save to apply to ReActor; no package changes or server restart required.",
                scale=2,
            )
            save_device_btn = gr.Button("Save", scale=0)
        save = gr.Markdown("")
        setattr(device, "do_not_save_to_config", True)
        save_device_btn.click(
            set_Device,
            inputs=[device],
            outputs=[save],
            queue=False,
            api_name=False,
        )
        with gr.Row():
            if len(models) == 0:
                logger.warning(
                    "You should at least have one model in models directory, please read the doc here: https://github.com/Gourieff/sd-webui-reactor/"
                )
                model = gr.Dropdown(
                    choices=models,
                    label="Model not found, please download one and refresh the list"
                )
            else:
                model = gr.Dropdown(
                    choices=models, label="Model", value=models[0]
                )
            models_update = gr.Button(
                value="🔄",
                variant="tool",
            )
            models_update.click(
                update_models_list, 
                inputs=[model],
                outputs=[model],
            )
            console_logging_level = gr.Radio(
                ["No log", "Minimum", "Default"],
                value="Minimum",
                label="Console Log Level",
                type="index"
            )
        gr.Markdown("<br>", visible=hash_check_block)
        with gr.Row(visible=hash_check_block):
            source_hash_check = gr.Checkbox(
                True,
                label="Source Image Hash Check",
                info="Recommended to keep it ON. Processing is faster when Source Image is the same."
            )
            target_hash_check = gr.Checkbox(
                False,
                label="Target Image Hash Check",
                info="Affects if you use Extras tab or img2img with only 'Swap in source image' on."
            )
    return model, device, console_logging_level, source_hash_check, target_hash_check
