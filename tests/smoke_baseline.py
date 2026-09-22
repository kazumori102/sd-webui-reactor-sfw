"""Explicit, real-model smoke test using the pre-ReActor package versions.

Run with Forge's Python: python -B tests/smoke_baseline.py
Models/downloads/output stay inside .test-artifacts. Existing inswapper is read
from the host models directory. No host package or configuration is modified.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path

from baseline_support import ROOT, setup_forge


def main():
    import numpy as np
    import insightface
    import onnxruntime as ort
    import onnx
    from PIL import Image

    versions = {"insightface": insightface.__version__, "onnxruntime": ort.__version__,
                "onnx": onnx.__version__, "numpy": np.__version__, "gradio": metadata.version("gradio")}
    assert versions == {"insightface": "1.0.1", "onnxruntime": "1.29.0", "onnx": "1.22.0", "numpy": "2.3.5", "gradio": "4.40.0"}, versions
    print("BASELINE", json.dumps(versions), flush=True)
    artifacts = ROOT / ".test-artifacts"
    artifacts.mkdir(exist_ok=True)
    original_cwd = Path.cwd()
    with setup_forge():
        import modules.paths_internal as paths
        paths.models_path = str(artifacts / "models")
        os.chdir(artifacts)
        try:
            from scripts import reactor_swapper as swapper
            from scripts.console_log_patch import apply_logging_patch
            apply_logging_patch(1)
            sample = Path(insightface.__file__).parent / "data/images/t1.jpg"
            with Image.open(sample) as image:
                target = image.convert("RGB")
            model = ROOT.parents[1] / "models/insightface/inswapper_128.onnx"
            assert model.is_file(), model
            result, info, swapped = swapper.swap_face(
                target, target, str(model), source_faces_index=[1], faces_index=[0],
                device="CPU", enhancement_options=swapper.EnhancementOptions(),
                detection_options=swapper.DetectionOptions(),
            )
            assert swapped == 1, (swapped, info)
            assert result.size == target.size
            assert np.any(np.asarray(result) != np.asarray(target)), "No pixels changed"
            from insightface.model_zoo.scrfd import SCRFD
            assert isinstance(swapper.ANALYSIS_MODEL.det_model, SCRFD)
            sessions = [model.session for model in swapper.ANALYSIS_MODEL.models.values()] + [swapper.FS_MODEL.session]
            assert all(session.get_providers() == ["CPUExecutionProvider"] for session in sessions)
            result.save(artifacts / "baseline-swap.png")
            print("PASS: real SFW check, SCRFD detection, face swap; all ORT sessions use CPU", flush=True)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    main()
