"""Run real Forge initialization, UI callbacks and ReActor inference.

Run from Forge's directory with its Python:
    python -B extensions/sd-webui-reactor-sfw/tests/smoke_forge.py

Uses installed packages (no .test-runtime or fake Forge modules). Downloads
missing inference models into Forge's models directory. Test settings and
results stay under .test-artifacts/forge; the user's settings are preserved.
"""
import asyncio
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
ARTIFACTS = ROOT / ".test-artifacts" / "forge"


def check_processing_hooks(target, sample):
    import gradio as gr
    from gradio.data_classes import FileData, ListFiles
    import numpy as np
    from modules import processing, scripts, scripts_postprocessing
    from scripts import reactor_helpers

    def value(control):
        if isinstance(control, (gr.Radio, gr.Dropdown)):
            return control.preprocess(control.value)
        return control.value

    def check_image(image, name):
        assert image.size == target.size
        assert np.any(np.asarray(image) != np.asarray(target)), name
        image.save(ARTIFACTS / f"{name}.png")

    common = dict(img=target, enable=True, source_faces_index="1", faces_index="0",
                  model="inswapper_128.onnx", device="CPU", selected_tab="tab_single",
                  select_source=0, face_restorer_name="None", upscaler_name="None",
                  mask_face=False)
    original_base = reactor_helpers.BASE_PATH
    reactor_helpers.BASE_PATH = str(ARTIFACTS)
    try:
        for name, runner, processing_type in (
            ("txt2img-hook", scripts.scripts_txt2img, processing.StableDiffusionProcessingTxt2Img),
            ("img2img-hook", scripts.scripts_img2img, processing.StableDiffusionProcessingImg2Img),
        ):
            script = next(s for s in runner.scripts if type(s).__name__ == "FaceSwapScript")
            controls = runner.inputs[script.args_from:script.args_to]
            names = list(inspect.signature(script.process).parameters)[1:]
            assert len(controls) == len(names)
            args = {key: value(control) for key, control in zip(names, controls)}
            args.update(common, swap_in_source=True, swap_in_generated=True, save_original=False)
            p = processing_type()
            if name == "img2img-hook":
                p.init_images = [target.copy()]
            script.process(p, **args)
            if name == "img2img-hook":
                check_image(p.init_images[0], "img2img-source")
            pp = scripts.PostprocessImageArgs(target.copy(), index=0)
            script.postprocess_image(p, pp)
            check_image(pp.image, name)
        print("PASS: real txt2img/img2img hooks, including Swap in Source", flush=True)

        extras = next(s for s in scripts.scripts_postproc.scripts if s.name == "ReActor")
        args = {key: value(control) for key, control in extras.controls.items()}
        args.update(common)
        pp = scripts_postprocessing.PostprocessedImage(target.copy())
        extras.process(pp, **args)
        assert pp.info.get("ReActor") is True
        check_image(pp.image, "extras-single")

        uploads = ListFiles(root=[FileData(path=str(sample)), FileData(path=str(sample))])
        args.update(selected_tab="tab_multiple", imgs=extras.controls["imgs"].preprocess(uploads))
        pp = scripts_postprocessing.PostprocessedImage(target.copy())
        extras.process(pp, **args)
        assert len(pp.extra_images) == 1, len(pp.extra_images)
        check_image(pp.image, "extras-multiple-1")
        check_image(pp.extra_images[0], "extras-multiple-2")
        print("PASS: real Extras single/multiple images with Gradio upload values", flush=True)
    finally:
        reactor_helpers.BASE_PATH = original_base


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    os.chdir(FORGE)
    sys.path.insert(0, str(FORGE))
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["SD_WEBUI_RESTARTING"] = "1"
    settings = json.loads((FORGE / "config.json").read_text(encoding="utf-8"))
    settings.update(clean_temp_dir_at_start=False, auto_launch_browser="Disable")
    settings_path = ARTIFACTS / "config.json"
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    ui_settings_path = ARTIFACTS / "ui-config.json"
    shutil.copyfile(FORGE / "ui-config.json", ui_settings_path)
    sys.argv = ["launch.py", "--skip-prepare-environment", "--api", "--xformers", "--sage",
                "--ui-settings-file", str(settings_path), "--ui-config-file", str(ui_settings_path)]
    # Match the portable launcher's external model directories when present.
    model_root = Path("G:/model")
    for flag, directory in (("--ckpt-dir", "CheckPoint"), ("--lora-dir", "Lora"),
                            ("--vae-dir", "vae"), ("--controlnet-dir", "controlnet"),
                            ("--embeddings-dir", "embeddings"), ("--esrgan-models-path", "ESRGAN"),
                            ("--codeformer-models-path", "Codeformer"), ("--gfpgan-models-path", "GFPGAN")):
        if (model_root / directory).is_dir():
            sys.argv.extend([flag, str(model_root / directory)])

    import webui
    from modules import extensions, script_callbacks, scripts, shared, ui
    import insightface
    import numpy as np
    import onnxruntime
    from PIL import Image

    active = [e.name for e in extensions.active() if "reactor" in e.name]
    assert active == ["sd-webui-reactor-sfw"], active
    script_callbacks.before_ui_callback()
    shared.demo = ui.create_ui()
    from scripts import reactor_helpers, reactor_swapper
    assert Path(reactor_swapper.__file__).resolve().is_relative_to(ROOT)
    print("PASS: real Forge initialization and UI; SFW extension loaded", flush=True)
    print("PACKAGES", insightface.__file__, onnxruntime.__file__, flush=True)

    # These are the actual txt2img, img2img and Extras Save events.
    events = [f for f in shared.demo.fns.values() if f.fn == reactor_helpers.set_Device]
    assert len(events) == 3, len(events)
    original_base = reactor_helpers.BASE_PATH
    reactor_helpers.BASE_PATH = str(ARTIFACTS)
    try:
        for event in events:
            response = asyncio.run(shared.demo.process_api(event, ["CPU"]))
            assert "CPU" in response["data"][0], response
    finally:
        reactor_helpers.BASE_PATH = original_base
    print("PASS: all three real Forge device Save callbacks", flush=True)

    sample = Path(insightface.__file__).parent / "data/images/t1.jpg"
    with Image.open(sample) as image:
        target = image.convert("RGB")
    result, info, swapped = reactor_swapper.swap_face(
        target, target, str(FORGE / "models/insightface/inswapper_128.onnx"),
        source_faces_index=[1], faces_index=[0], device="CPU",
        enhancement_options=reactor_swapper.EnhancementOptions(),
        detection_options=reactor_swapper.DetectionOptions(),
    )
    assert swapped == 1, (swapped, info)
    assert result.size == target.size
    assert np.any(np.asarray(result) != np.asarray(target)), "No pixels changed"
    result.save(ARTIFACTS / "swap.png")
    sessions = [m.session for m in reactor_swapper.ANALYSIS_MODEL.models.values()]
    sessions.append(reactor_swapper.FS_MODEL.session)
    assert all(s.get_providers() == ["CPUExecutionProvider"] for s in sessions)
    print("PASS: real SFW classification, detection and face swap in Forge", flush=True)

    check_processing_hooks(target, sample)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from modules.api import api
    app = FastAPI()
    script_callbacks.app_started_callback(shared.demo, app)
    with TestClient(app) as client:
        response = client.get("/reactor/models")
        assert response.status_code == 200, response.text
        assert "inswapper_128.onnx" in response.json()["models"], response.text
        encoded = api.encode_pil_to_base64(target).decode("ascii")
        response = client.post("/reactor/image", json={
            "source_image": encoded, "target_image": encoded,
            "source_faces_index": [1], "face_index": [0], "device": "CPU",
        })
        assert response.status_code == 200, response.text[:500]
        api_image = api.decode_base64_to_image(response.json()["image"])
        assert api_image.size == target.size
        assert np.any(np.asarray(api_image) != np.asarray(target))
        api_image.save(ARTIFACTS / "api-swap.png")
    app.state.executor.shutdown(wait=True)
    print("PASS: real /reactor/models and /reactor/image endpoints", flush=True)


if __name__ == "__main__":
    main()
