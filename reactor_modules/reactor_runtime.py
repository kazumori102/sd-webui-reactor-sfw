"""ReActor-owned runtime state; no patches to shared libraries."""
from functools import wraps
from pathlib import Path
from threading import RLock

import insightface
import onnxruntime
from insightface.app import FaceAnalysis


_lock = RLock()


def serialized(function):
    """Protect ReActor's model and face caches across WebUI/API requests."""
    @wraps(function)
    def call(*args, **kwargs):
        with _lock:
            return function(*args, **kwargs)
    return call


def get_available_devices():
    providers = onnxruntime.get_available_providers()
    return [device for device, provider in (("CPU", "CPUExecutionProvider"), ("CUDA", "CUDAExecutionProvider")) if provider in providers]


def get_providers(device):
    if device not in get_available_devices():
        raise ValueError(f"{device!r} is unavailable in the installed ONNX Runtime. Available devices: {get_available_devices()}")
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "CUDA" else ["CPUExecutionProvider"]


class ReActorFaceAnalysis(FaceAnalysis):
    """Use installed InsightFace inference/prepare without its global ORT log reset.

The upstream constructor calls set_default_logger_severity for the entire
process. Only model discovery is local here; routing, prepare and inference
remain the installed package's implementations (including 1.0.1 SCRFD).
"""
    def __init__(self, name="buffalo_l", root="~/.insightface", providers=None):
        self.models = {}
        self.model_dir = insightface.utils.ensure_available("models", name, root=root)
        for path in sorted(Path(self.model_dir).glob("*.onnx")):
            model = insightface.model_zoo.get_model(str(path), providers=providers)
            if model is not None and model.taskname not in self.models:
                self.models[model.taskname] = model
        if "detection" not in self.models:
            raise RuntimeError(f"No detection model found in {self.model_dir}")
        self.det_model = self.models["detection"]
        self._reactor_prepare_key = None

    def prepare(self, ctx_id, det_thresh=0.5, det_size=None):
        key = (ctx_id, det_thresh, repr(det_size))
        if self._reactor_prepare_key == key:
            return
        super().prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
        self._reactor_prepare_key = key
