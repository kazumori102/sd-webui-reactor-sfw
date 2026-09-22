import importlib
import logging
import sys
import unittest
import warnings
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baseline_support import ROOT, setup_forge


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = setup_forge()
        cls.addClassCleanup(cls.directory.cleanup)
        import insightface
        import onnxruntime
        if insightface.__version__ != "1.0.1" or onnxruntime.__version__ != "1.29.0":
            raise RuntimeError("Tests require InsightFace 1.0.1 / ORT 1.29.0; use Forge's Python")
        from scripts import reactor_helpers
        cls.helpers = reactor_helpers

    def test_logging_keeps_insightface_101_classes_intact(self):
        from insightface.app import FaceAnalysis
        from insightface.model_zoo.model_zoo import ModelRouter
        from insightface.model_zoo.inswapper import INSwapper
        from scripts.console_log_patch import apply_logging_patch
        originals = (FaceAnalysis.__init__, FaceAnalysis.prepare, ModelRouter.get_model, INSwapper.__init__)
        for level in (0, 1, 2):
            apply_logging_patch(level)
            self.assertEqual((FaceAnalysis.__init__, FaceAnalysis.prepare, ModelRouter.get_model, INSwapper.__init__), originals)

    def test_invalid_device_does_not_overwrite_saved_setting(self):
        with patch.object(self.helpers, "BASE_PATH", self.directory.name):
            self.helpers.set_Device("CPU")
            for value in (None, "", "bogus"):
                with self.assertRaises(ValueError):
                    self.helpers.set_Device(value)
            self.assertEqual(self.helpers.get_Device(), "CPU")

    def test_startup_does_not_download_models_or_modify_global_filters(self):
        # Import third-party dependencies first; measure only ReActor's effects.
        import numpy as np
        from transformers import pipeline
        from scipy import stats
        import reactor_modules.reactor_mask
        before = list(warnings.filters)
        transformers_level = logging.getLogger("transformers").level
        with patch("urllib.request.urlopen", side_effect=AssertionError("startup download")):
            importlib.reload(importlib.import_module("scripts.reactor_sfw"))
            importlib.reload(importlib.import_module("scripts.reactor_swapper"))
        self.assertEqual(warnings.filters, before)
        self.assertFalse(hasattr(np, "warnings"))
        self.assertEqual(logging.getLogger("transformers").level, transformers_level)

    def test_gradio4_refresh_preserves_only_existing_choices(self):
        from reactor_ui import ui_settings, ui_upscale, ui_main
        with patch.object(ui_settings, "get_models", return_value=["a.onnx"]):
            self.assertEqual(ui_settings.update_models_list("removed.onnx")["value"], "a.onnx")
        self.assertEqual(ui_upscale.update_upscalers_list("None")["value"], "None")
        with patch.object(ui_main, "get_model_names", return_value=["None", "face.safetensors"]):
            self.assertEqual(ui_main.update_fm_list("removed.safetensors")["value"], "None")

    def test_gradio4_source_selector_returns_supported_updates(self):
        import gradio as gr
        from reactor_ui import ui_main
        with gr.Blocks(analytics_enabled=False) as demo:
            ui_main.show(False, extra_multiple_source="")
        event = next(f for f in demo.fns.values() if f.fn.__name__ == "on_select_source")
        for index, expected in ((0, [True, False, False, False]),
                                (1, [False, True, False, False]),
                                (2, [False, False, True, True])):
            data = gr.SelectData(None, {"index": index, "value": "", "selected": True})
            result = asyncio.run(demo.process_api(event, [], event_data=data))
            self.assertEqual([update["visible"] for update in result["data"]], expected)

    def test_gradio4_switching_back_to_single_selects_single_image(self):
        import gradio as gr
        from reactor_ui import ui_main
        with gr.Blocks(analytics_enabled=False) as demo:
            ui_main.show(False, extra_multiple_source="")
        for label, expected in (("Multiple", "tab_multiple"), ("Single", "tab_single")):
            tab = next(c for c in demo.blocks.values() if isinstance(c, gr.Tab) and c.label == label)
            events = [f for f in demo.fns.values() if (tab._id, "select") in f.targets]
            self.assertEqual(len(events), 1, f"{label} tab must update source selection")
            result = asyncio.run(demo.process_api(events[0], []))
            self.assertEqual(result["data"], [expected])

    def test_sfw_block_in_multiple_source_mode_returns_empty_results(self):
        from scripts import reactor_swapper as swapper
        from PIL import Image
        target = Image.new("RGB", (16, 16))
        with patch.object(swapper, "check_sfw_image", return_value=None):
            for args in ({"source_imgs": ["unused.png"]}, {"select_source": 2, "source_folder": "unused"}):
                result, info, count = swapper.swap_face(None, target, **args)
                self.assertEqual(result, [])
                self.assertEqual(count, 0)

    def test_empty_source_folder_returns_empty_results(self):
        from scripts import reactor_swapper as swapper
        from PIL import Image
        import tempfile
        target = Image.new("RGB", (16, 16))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(swapper, "check_sfw_image", return_value=target):
            for random_image in (False, True):
                with self.subTest(random_image=random_image):
                    result, info, count = swapper.swap_face(
                        None, target, "unused.onnx", select_source=2, source_folder=directory,
                        random_image=random_image, enhancement_options=swapper.EnhancementOptions(),
                        detection_options=swapper.DetectionOptions(),
                    )
                    self.assertEqual(result, [])
                    self.assertEqual(count, 0)

    def test_gradio4_device_events_have_independent_inputs(self):
        import gradio as gr
        from reactor_ui import ui_settings
        with patch.object(self.helpers, "BASE_PATH", self.directory.name):
            with gr.Blocks(analytics_enabled=False) as demo:
                for _ in range(3):
                    ui_settings.show()
            events = [f for f in demo.fns.values() if f.fn == self.helpers.set_Device]
            self.assertEqual(len(events), 3)
            self.assertEqual(len({f.inputs[0]._id for f in events}), 3)
            for event in events:
                result = asyncio.run(demo.process_api(event, ["CPU"]))
                self.assertIn("CPU", result["data"][0])
            self.assertEqual(Path(self.directory.name, "last_device.txt").read_text(), "CPU")

    def test_model_load_failure_can_be_retried(self):
        from scripts import reactor_swapper as swapper
        old = (swapper.FS_MODEL, swapper.CURRENT_FS_MODEL_PATH)
        try:
            swapper.FS_MODEL = swapper.CURRENT_FS_MODEL_PATH = None
            loaded = object()
            with patch.object(swapper.insightface.model_zoo, "get_model", side_effect=[RuntimeError("bad model"), loaded]):
                with self.assertRaises(RuntimeError):
                    swapper.getFaceSwapModel("retry.onnx")
                self.assertIs(swapper.getFaceSwapModel("retry.onnx"), loaded)
        finally:
            swapper.FS_MODEL, swapper.CURRENT_FS_MODEL_PATH = old

    def test_analysis_does_not_deepcopy_ort_sessions(self):
        from scripts import reactor_swapper as swapper
        class Analysis:
            def __deepcopy__(self, memo):
                raise AssertionError("deepcopy recreates ORT sessions without providers")
            def prepare(self, **kwargs):
                self.context = kwargs["ctx_id"]
            def get(self, img, max_num):
                return [self.context]
        with patch.object(swapper, "getAnalysisModel", return_value=Analysis()), \
             patch.object(swapper, "PROVIDERS", ["CPUExecutionProvider"]):
            self.assertEqual(swapper.analyze_faces(None), [-1])

    def test_sfw_checks_nsfw_label_not_highest_class(self):
        from scripts import reactor_sfw as sfw
        from PIL import Image
        class Classifier:
            model = SimpleNamespace(to=lambda device: None)
            def __call__(self, image, **kwargs):
                return [{"label": "sfw", "score": .999}, {"label": "nsfw", "score": .001}]
        path = Path(self.directory.name, "safe.png")
        Image.new("RGB", (16, 16)).save(path)
        with patch.object(sfw, "pipeline", return_value=Classifier()):
            self.assertFalse(sfw.nsfw_image(str(path), self.directory.name))

    def test_ort_capabilities_determine_available_devices(self):
        from reactor_modules.reactor_runtime import get_available_devices, get_providers
        self.assertEqual(get_available_devices(), ["CPU"])
        self.assertEqual(get_providers("CPU"), ["CPUExecutionProvider"])
        with self.assertRaises(ValueError):
            get_providers("CUDA")
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            self.assertEqual(get_providers("CUDA"), ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_analysis_preserves_101_prepare_without_global_ort_changes(self):
        from reactor_modules.reactor_runtime import ReActorFaceAnalysis
        from unittest.mock import Mock
        model_dir = Path(self.directory.name, "pack")
        model_dir.mkdir(exist_ok=True)
        (model_dir / "detector.onnx").touch()
        detector = SimpleNamespace(taskname="detection", prepare=Mock())
        with patch("insightface.utils.ensure_available", return_value=str(model_dir)), \
             patch("insightface.model_zoo.get_model", return_value=detector), \
             patch("onnxruntime.set_default_logger_severity", side_effect=AssertionError("global ORT change")):
            analysis = ReActorFaceAnalysis(root=self.directory.name, providers=["CPUExecutionProvider"])
            analysis.prepare(ctx_id=-1, det_size=None)
        self.assertIs(analysis.det_model, detector)
        self.assertEqual(detector.prepare.call_args.kwargs["input_size"], [(128, 128), (640, 640)])


if __name__ == "__main__":
    unittest.main()
