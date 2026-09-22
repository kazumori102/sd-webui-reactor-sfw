"""Exercise installer decisions without changing the test runner's packages."""
import contextlib
import importlib.metadata as metadata
import io
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    "insightface": "1.0.1", "onnx": "1.22.0", "onnxruntime": "1.29.0",
    "numpy": "2.3.5", "opencv-python": "5.0.0.93", "tqdm": "4.67.1",
    "requests": "2.32.5", "scipy": "1.17.0", "scikit-image": "0.26.0",
    "torch": "2.13.0+cu130", "gradio": "4.40.0",
}


class InstallerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="reactor-installer-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.extension = self.root / "extension"
        self.extension.mkdir()
        for name in ("install.py", "requirements.txt"):
            shutil.copyfile(ROOT / name, self.extension / name)
        self.model = self.root / "models/insightface/inswapper_128.onnx"
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"existing model")
        self.versions = dict(BASELINE)
        self.calls = []
        self.constraints = []
        self.cuda = "13.0"
        self.providers = ["CPUExecutionProvider"]
        self.fail_command = None
        self.fail_install_once = False

    def pip(self, command, **kwargs):
        self.assertEqual(command[:3], [sys.executable, "-m", "pip"])
        self.assertTrue(kwargs.get("check"), "pip failure must stop the installer")
        self.calls.append(command)
        action = command[3]
        if self.fail_command == action:
            raise subprocess.CalledProcessError(1, command)
        if self.fail_install_once and action == "install":
            self.fail_install_once = False
            raise subprocess.CalledProcessError(1, command)
        if "--constraint" in command:
            self.constraints.append(Path(command[command.index("--constraint") + 1]).read_text())
        if action == "download":
            destination = Path(command[command.index("--dest") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            req = Requirement(command[-1])
            version = next((s.version for s in req.specifier if s.operator in ("==", ">=")), "1.29.0")
            (destination / f"{req.name.replace('-', '_')}-{version}-cp313-cp313-win_amd64.whl").write_bytes(b"wheel")
        elif action == "uninstall":
            for name in command[4:]:
                if not name.startswith("-"):
                    self.versions.pop(name, None)
        elif action == "install":
            # Installation is external; simulate only its package metadata effects.
            for argument in command[4:]:
                if argument.endswith(".whl"):
                    name, version = Path(argument).name.split("-")[:2]
                    self.versions[name.replace("_", "-")] = version
                elif argument.startswith(("insightface", "onnx", "numpy", "opencv", "tqdm", "requests", "scipy", "scikit-image")):
                    req = Requirement(argument)
                    self.versions[req.name] = BASELINE.get(req.name, "1.29.0")
        return subprocess.CompletedProcess(command, 0)

    def run_installer(self, *args, response=None):
        def version(name):
            if name not in self.versions:
                raise metadata.PackageNotFoundError(name)
            return self.versions[name]

        distributions = [types.SimpleNamespace(metadata={"Name": k}, version=v)
                         for k, v in self.versions.items()]
        paths = types.ModuleType("modules.paths_internal")
        paths.models_path = str(self.root / "models")
        torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=self.cuda),
                                      cuda=types.SimpleNamespace(is_available=lambda: self.cuda is not None))
        ort = types.SimpleNamespace(get_available_providers=lambda: self.providers)
        capture = io.StringIO()
        with patch.dict(sys.modules, {"modules.paths_internal": paths, "torch": torch, "onnxruntime": ort}), \
             patch.object(sys, "argv", [str(self.extension / "install.py"), *args]), \
             patch.object(metadata, "version", side_effect=version), \
             patch.object(metadata, "distributions", return_value=distributions), \
             patch.object(subprocess, "run", side_effect=self.pip), \
             patch("urllib.request.urlopen", return_value=response,
                   side_effect=None if response is not None else AssertionError("unexpected network request")), \
             contextlib.redirect_stdout(capture):
            runpy.run_path(str(self.extension / "install.py"), run_name="__main__")
        return capture.getvalue()

    def test_satisfied_environment_is_unchanged_on_repeated_startup(self):
        self.run_installer()
        self.run_installer()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.model.read_bytes(), b"existing model")
        self.assertEqual((self.extension / "last_device.txt").read_text().strip(), "CPU")

    def test_only_missing_and_old_requirements_are_installed(self):
        self.versions["insightface"] = "0.7.3"
        del self.versions["onnx"]
        self.run_installer()
        self.assertEqual(self.versions["insightface"], "1.0.1")
        self.assertEqual(self.versions["onnx"], "1.22.0")
        self.assertEqual(self.versions["numpy"], "2.3.5")
        self.assertTrue(any("--no-deps" in c and any("insightface" in a for a in c) for c in self.calls))
        self.assertTrue(any("numpy==2.3.5" in c and "torch==2.13.0+cu130" in c for c in self.constraints))

    def test_newer_versions_and_gpu_headless_variants_are_preserved(self):
        self.versions.update(insightface="1.0.2", onnx="1.23.0")
        del self.versions["onnxruntime"]
        del self.versions["opencv-python"]
        self.versions.update({"onnxruntime-gpu": "1.29.0", "opencv-python-headless": "5.0.0.93"})
        self.providers.append("CUDAExecutionProvider")
        self.run_installer()
        self.assertEqual(self.calls, [])
        self.assertEqual((self.extension / "last_device.txt").read_text().strip(), "CUDA")

    def test_missing_runtime_uses_cuda_13_wheel_and_keeps_saved_cpu_selection(self):
        del self.versions["onnxruntime"]
        self.providers.append("CUDAExecutionProvider")
        (self.extension / "last_device.txt").write_text("CPU")
        self.run_installer()
        self.assertIn("onnxruntime-gpu", self.versions)
        self.assertNotIn("onnxruntime", self.versions)
        self.assertEqual((self.extension / "last_device.txt").read_text().strip(), "CPU")

    def test_gpu_switch_downloads_before_uninstalling_cpu(self):
        self.providers.append("CUDAExecutionProvider")
        self.run_installer("--device", "CUDA")
        actions = [c[3] for c in self.calls]
        self.assertLess(actions.index("download"), actions.index("uninstall"))
        self.assertLess(actions.index("uninstall"), actions.index("install"))
        self.assertNotIn("onnxruntime", self.versions)
        self.assertEqual(self.versions["onnxruntime-gpu"], "1.29.0")
        self.assertEqual((self.extension / "last_device.txt").read_text().strip(), "CUDA")

    def test_download_failure_keeps_existing_runtime_and_device(self):
        self.fail_command = "download"
        (self.extension / "last_device.txt").write_text("CPU")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_installer("--device", "CUDA")
        self.assertEqual(self.versions["onnxruntime"], "1.29.0")
        self.assertFalse(any(c[3] == "uninstall" for c in self.calls))
        self.assertEqual((self.extension / "last_device.txt").read_text(), "CPU")

    def test_pip_failure_stops_before_downloading_model(self):
        del self.versions["insightface"]
        self.model.unlink()
        self.fail_command = "install"
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_installer()
        self.assertFalse(self.model.exists())

    def test_failed_gpu_install_restores_cpu_wheel(self):
        self.fail_install_once = True
        (self.extension / "last_device.txt").write_text("CPU")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_installer("--device", "CUDA")
        self.assertEqual(self.versions["onnxruntime"], "1.29.0")
        self.assertNotIn("onnxruntime-gpu", self.versions)
        self.assertEqual((self.extension / "last_device.txt").read_text(), "CPU")

    def test_runtime_install_does_not_force_reinstall_shared_dependencies(self):
        self.providers.append("CUDAExecutionProvider")
        self.run_installer("--device", "CUDA")
        for command in self.calls:
            if command[3] == "install" and "--no-deps" not in command:
                self.assertNotIn("--force-reinstall", command)

    def test_cpu_only_host_installs_cpu_runtime(self):
        self.cuda = None
        del self.versions["onnxruntime"]
        self.run_installer()
        self.assertEqual(self.versions["onnxruntime"], "1.29.0")
        self.assertNotIn("onnxruntime-gpu", self.versions)

    def test_cuda_12_does_not_install_cuda_13_runtime(self):
        self.cuda = "12.10"
        self.providers.append("CUDAExecutionProvider")
        self.run_installer("--device", "CUDA")
        specs = [Requirement(c[-1]) for c in self.calls if c[3] == "download" and "onnxruntime-gpu" in c[-1]]
        self.assertTrue(specs)
        self.assertTrue(specs[0].specifier.contains("1.26.0"))
        self.assertFalse(specs[0].specifier.contains("1.29.0"))

    def test_invalid_saved_device_recovers_without_changing_packages(self):
        (self.extension / "last_device.txt").write_text("bogus")
        self.run_installer()
        self.assertEqual((self.extension / "last_device.txt").read_text(), "CPU")
        self.assertEqual(self.calls, [])

    def test_missing_model_is_downloaded_and_reused(self):
        self.model.unlink()
        response = io.BytesIO(b"complete model")
        response.headers = {"Content-Length": "14"}
        self.run_installer(response=response)
        self.assertEqual(self.model.read_bytes(), b"complete model")
        self.assertEqual(list(self.model.parent.glob("*.part")), [])
        self.run_installer()

    def test_short_download_does_not_publish_a_broken_model(self):
        self.model.write_bytes(b"")
        response = io.BytesIO(b"partial")
        response.headers = {"Content-Length": "100"}
        with self.assertRaises((OSError, RuntimeError)):
            self.run_installer(response=response)
        self.assertEqual(self.model.read_bytes(), b"")
        self.assertEqual(list(self.model.parent.glob("*.part")), [])

    def test_dry_run_lists_gpu_migration_and_missing_model_without_writes(self):
        self.versions["insightface"] = "0.7.3"
        self.model.unlink()
        before = {str(p.relative_to(self.root)): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        with patch.object(tempfile, "TemporaryDirectory", side_effect=AssertionError("dry-run created temp directory")), \
             patch.object(tempfile, "NamedTemporaryFile", side_effect=AssertionError("dry-run created temp file")):
            output = self.run_installer("--dry-run", "--device", "CUDA")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.versions["insightface"], "0.7.3")
        self.assertEqual(self.versions["onnxruntime"], "1.29.0")
        after = {str(p.relative_to(self.root)): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        self.assertIn("insightface>=1.0.1", output)
        self.assertIn("Download wheel: onnxruntime-gpu", output)
        self.assertIn("Uninstall: onnxruntime", output)
        self.assertIn("Download model:", output)
        self.assertIn("Execution Provider: CUDA", output)

    def test_dryrun_alias_preserves_existing_cpu_selection(self):
        (self.extension / "last_device.txt").write_text("CPU")
        output = self.run_installer("--dryrun")
        self.assertEqual(self.calls, [])
        self.assertIn("Keep runtime: onnxruntime 1.29.0", output)
        self.assertIn("Keep model:", output)
        self.assertIn("Execution Provider: CPU", output)
        self.assertEqual((self.extension / "last_device.txt").read_text(), "CPU")


if __name__ == "__main__":
    unittest.main()
