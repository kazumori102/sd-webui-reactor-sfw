"""Test the host packages without modifying the host installation.

Only Forge services are replaced: these tests run outside the live WebUI.
Gradio, InsightFace, NumPy, ONNX, ORT and PyTorch are the real packages.
"""
from pathlib import Path
import os
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("REACTOR_TEST_RUNTIME"):
    sys.path.insert(0, os.environ["REACTOR_TEST_RUNTIME"])


def module(name, **members):
    result = types.ModuleType(name)
    result.__dict__.update(members)
    sys.modules[name] = result
    return result


def setup_forge():
    directory = tempfile.TemporaryDirectory(prefix="reactor-test-")
    modules = module("modules", __path__=[])
    shared = module("modules.shared", cmd_opts=types.SimpleNamespace(),
                    opts=types.SimpleNamespace(data={}), device="cpu",
                    state=types.SimpleNamespace(interrupted=False, skipped=False),
                    sd_upscalers=[types.SimpleNamespace(name="None")],
                    face_restorers=[])
    modules.shared = shared
    module("modules.paths_internal", models_path=directory.name)
    module("modules.images", FilenameGenerator=object, get_next_sequence_number=lambda *a: 0)
    module("modules.script_callbacks")
    module("modules.face_restoration", FaceRestoration=object)
    module("modules.upscaler", UpscalerData=object)
    module("modules.codeformer_model")
    module("modules.gfpgan_model")
    return directory
