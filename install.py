"""Install ReActor dependencies and its swap model for the current Forge Python.

Normal startup preserves an existing ORT edition. Use --device CUDA to migrate
to a GPU edition compatible with PyTorch, or --device CPU to select CPU inference.
Add --dry-run to preview the changes without pip, network access or file writes.
"""
import subprocess
import os, sys
import argparse
from importlib import metadata
from pathlib import Path
import tempfile
import urllib.request
from packaging.version import Version as pv
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


if "--dry-run" in sys.argv or "--dryrun" in sys.argv:
    sys.dont_write_bytecode = True

BASE_PATH = Path(__file__).resolve().parent

req_file = BASE_PATH / "requirements.txt"

ORT_PACKAGES = ("onnxruntime-gpu", "onnxruntime", "onnxruntime-directml")
OPENCV_PACKAGES = ("opencv-python", "opencv-python-headless",
                   "opencv-contrib-python", "opencv-contrib-python-headless")

model_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
model_name = Path(model_url).name


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


# Keep the original installer API as the main vocabulary.  The implementation
# below adds constraints and rollback around these operations.
def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])


def pip_uninstall(*args):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])


def is_installed(
		package: str, required=None, strict=True
):

    current = installed_version(package)
    if current is None:
        return False

    if required is None:
        return True

    requirement = Requirement(f"{package}{'==' if strict else '>='}{required}")
    return requirement.specifier.contains(current, prereleases=True)


def requirements():
    result = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.marker is None or req.marker.evaluate():
            result.append(req)
    return result


def missing_requirements():
    missing = []
    for req in requirements():
        names = OPENCV_PACKAGES if canonicalize_name(req.name) == "opencv-python" else (req.name,)
        present = [(name, installed_version(name)) for name in names]
        matching = [(name, version) for name, version in present
                    if version is not None and req.specifier.contains(version, prereleases=True)]
        if matching:
            name, version = matching[0]
            print(f"[ReActor] Found {name} {version}")
        else:
            # Upgrade the existing OpenCV edition instead of installing a second cv2.
            existing = next((name for name, version in present if version is not None), req.name)
            missing.append(Requirement(f"{existing}{req.specifier}"))
    return missing


def cuda_version():
    # Do not import ORT before changing its wheel: Windows locks loaded DLLs.
    import torch
    return torch.version.cuda if torch.cuda.is_available() else None


def runtime_requirement(device, cuda, installed):
    if device == "CUDA":
        name = "onnxruntime-gpu"
    elif installed:
        name = next(name for name in ORT_PACKAGES if name in installed)
    else:
        name = "onnxruntime-gpu" if device != "CPU" and cuda is not None else "onnxruntime"
    if name == "onnxruntime-gpu":
        major = pv(cuda).major if cuda else None
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        if major == 13:
            return Requirement("onnxruntime-gpu>=1.29.0,<1.30")
        if major == 12:
            return Requirement("onnxruntime-gpu>=1.21.0,<1.27")
        if name in installed and device != "CUDA":
            return Requirement(f"{name}>={installed[name]}")
        raise RuntimeError(f"Automatic GPU setup requires PyTorch CUDA 12 or 13; found {cuda!r}.")
    return Requirement(f"{name}>=1.29.0")


def write_constraints(path, updating):
    # Let pip add dependencies, but keep unrelated packages at their installed
    # versions. Incompatible requests fail instead of silently changing Forge.
    updating = {canonicalize_name(name) for name in updating}
    lines = set()
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name and canonicalize_name(name) not in updating:
            lines.add(f"{name}=={dist.version}")
    path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")


def download_wheel(requirement, directory):
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--dest", directory, requirement])
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel for {requirement}; found {len(wheels)}")
    return wheels[0]


def download(url, path):
    """Download one file atomically, retaining the old install.py API."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
             tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            expected = int(response.headers.get("Content-Length", 0))
            size = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                size += len(chunk)
        if size == 0 or (expected and size != expected):
            raise OSError(f"Incomplete download: received {size} bytes; expected {expected}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def runtime_update_needed(req, installed):
    current = installed.get(req.name)
    if current and req.specifier.contains(current, prereleases=True) and len(installed) == 1:
        return False
    # A newer GPU build can target a different CUDA major. Do not downgrade it.
    if current:
        minimum = next(pv(s.version) for s in req.specifier if s.operator == ">=")
        if pv(current) >= minimum and not req.specifier.contains(current, prereleases=True):
            raise RuntimeError(f"{req.name} {current} does not match {req}; select a matching PyTorch/ORT build.")
    return True


def install_runtime(req, installed, directory, constraints):
    if not runtime_update_needed(req, installed):
        print(f"[ReActor] Found {req.name} {installed[req.name]}")
        return 0
    wheel = download_wheel(str(req), directory / "new")
    # Fetch rollback wheels before uninstalling shared ORT files.
    previous = [download_wheel(f"{name}=={version}", directory / name)
                for name, version in installed.items()]
    if installed:
        pip_uninstall(*installed)
    try:
        pip_install("--constraint", constraints, "--upgrade-strategy", "only-if-needed", wheel)
    except subprocess.CalledProcessError:
        if previous:
            print("[ReActor] ORT installation failed; restoring the previous wheel(s).")
            pip_install("--no-deps", "--force-reinstall", *previous)
        raise
    return 1


def plan_dependencies(device):
    missing = missing_requirements()
    cuda = cuda_version()
    installed = {name: version for name in ORT_PACKAGES if (version := installed_version(name))}
    runtime = runtime_requirement(device, cuda, installed)
    runtime_update_needed(runtime, installed)  # Validate before changing any package.
    return missing, cuda, installed, runtime


def install_dependencies(device):
    missing, cuda, installed, runtime = plan_dependencies(device)
    with tempfile.TemporaryDirectory(prefix="reactor-install-") as directory:
        directory = Path(directory)
        constraints = directory / "constraints.txt"
        write_constraints(constraints, [req.name for req in missing] + list(ORT_PACKAGES))
        # Install InsightFace's inference dependencies explicitly. Its metadata
        # requires the CPU ORT and GUI OpenCV distribution names even when their
        # GPU/headless equivalents are already installed.
        ordinary = [str(req) for req in missing if canonicalize_name(req.name) != "insightface"]
        if ordinary:
            pip_install("--constraint", constraints, "--upgrade-strategy", "only-if-needed", *ordinary)
        count = len(ordinary) + install_runtime(runtime, installed, directory, constraints)
        for req in missing:
            if canonicalize_name(req.name) == "insightface":
                pip_install("--no-deps", str(req))
                count += 1
    remaining = missing_requirements()
    runtime_version = installed_version(runtime.name)
    if remaining or runtime_version is None or not runtime.specifier.contains(runtime_version, prereleases=True):
        raise RuntimeError("Dependency installation did not satisfy the requirements; see pip output above.")
    return count, cuda


def model_path():
    try:
        from modules.paths_internal import models_path
    except ImportError:
        try:
            from modules.paths import models_path
        except ImportError:
            models_path = Path.cwd() / "models"

    return Path(models_path) / "insightface/inswapper_128.onnx"


def ensure_model():
    path = model_path()
    if path.is_file() and path.stat().st_size > 0:
        print(f"[ReActor] Found model: {path}")
        return

    print(f"[ReActor] Downloading {model_name}...")
    download(model_url, path)
    print(f"[ReActor] Model ready: {path}")


def select_device(requested, available):
    if not available:
        raise RuntimeError("No supported ONNX Runtime execution provider is available.")

    if requested != "auto" and requested not in available:
        raise RuntimeError(f"Requested {requested} is unavailable after setup: {available}")

    path = BASE_PATH / "last_device.txt"
    saved = path.read_text(encoding="utf-8").strip() if path.exists() else None
    selected = requested if requested != "auto" else saved if saved in available else available[-1]
    return selected, saved


def save_device(requested, cuda):

    # All package changes have finished; it is now safe to load the runtime.
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    available = ["CPU"] if "CPUExecutionProvider" in providers else []
    if cuda and "CUDAExecutionProvider" in providers:
        available.append("CUDA")

    selected, saved = select_device(requested, available)
    if selected != saved:
        (BASE_PATH / "last_device.txt").write_text(selected, encoding="utf-8")

    print(f"[ReActor] Execution Provider: {selected}")


def dry_run(device):
    print("[ReActor] DRY RUN: no pip commands, downloads or file writes.")
    missing, cuda, installed, runtime = plan_dependencies(device)
    for req in missing:
        options = " (--no-deps; inference dependencies listed separately)" if req.name == "insightface" else ""
        print(f"[DRY-RUN] Install: {req}{options}")

    if runtime_update_needed(runtime, installed):
        print(f"[DRY-RUN] Download wheel: {runtime}")
        for name, version in installed.items():
                    print(f"[DRY-RUN] Download rollback wheel: {name}=={version}")

        if installed:
            print(f"[DRY-RUN] Uninstall: {', '.join(installed)} (after all wheels are downloaded)")

        print(f"[DRY-RUN] Install downloaded wheel: {runtime}")

    else:
        print(f"[DRY-RUN] Keep runtime: {runtime.name} {installed[runtime.name]}")

    print("[DRY-RUN] Other installed packages remain constrained to their current versions.")
    path = model_path()
    action = "Keep model" if path.is_file() and path.stat().st_size > 0 else "Download model"
    print(f"[DRY-RUN] {action}: {path}")
    available = ["CPU", "CUDA"] if runtime.name == "onnxruntime-gpu" and cuda else ["CPU"]
    selected, saved = select_device(device, available)
    print(f"[DRY-RUN] Execution Provider: {selected} (saved: {saved or 'unset'})")
    if selected != saved:
        print(f"[DRY-RUN] Save device to: {BASE_PATH / 'last_device.txt'}")

    print("[DRY-RUN] Wheel availability and dependency resolution are checked only during execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "CPU", "CUDA"), default="auto",
                        help="auto preserves the installed ORT edition and saved inference device")
    parser.add_argument("--dry-run", "--dryrun", action="store_true",
                        help="show planned changes without running pip, downloading or writing files")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.device)
        return
    install_count, cuda = install_dependencies(args.device)
    ensure_model()
    save_device(args.device, cuda)
    if install_count:
        print(f"""
        [ReActor] Dependencies updated. Restart any already running Forge process.
        +---------------------------------+
        --- PLEASE, RESTART the Server! ---
        +---------------------------------+
        """)
    else:
        print("[ReActor] Setup complete; existing packages were preserved.")


if __name__ == "__main__":
    main()
