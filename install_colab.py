#!/usr/bin/env python3
"""Migrate the Colab installation and print every installer error to the cell."""
import importlib.metadata
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def run(command):
    print("+ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    if sys.version_info < (3, 13):
        raise RuntimeError("Python 3.13+ is required; use the intended Colab interpreter")
    root = Path(__file__).resolve().parent
    pip = [sys.executable, "-m", "pip"]
    # Freeze existing accelerator builds, including +cu130 local version tags.
    constraints = []
    for name in ("torch", "torchvision"):
        try:
            constraints.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    print("Preserving installed accelerator packages: " + ", ".join(constraints), flush=True)
    run([*pip, "uninstall", "-y", "mediapipe"])
    # Only remove the two cached weights that this application downloaded.
    for name in ("blaze_face_short_range.tflite", "pose_landmarker_lite.task"):
        path = Path.home() / ".cache" / "mediapipe" / name
        if path.is_file():
            path.unlink()
            print(f"Removed old model: {path}", flush=True)
    # These distributions share cv2 files: reinstall one provider after removal.
    run([*pip, "uninstall", "-y", "opencv-python", "opencv-python-headless",
         "opencv-contrib-python", "opencv-contrib-python-headless"])
    with tempfile.TemporaryDirectory(prefix="reframe-install-") as directory:
        constraint_file = Path(directory) / "constraints.txt"
        constraint_file.write_text("\n".join(constraints) + "\n")
        run([*pip, "install", "--constraint", str(constraint_file),
             "--requirement", str(root / "requirements.txt")])
    # Install the complete package and console entry point. Editable mode makes
    # later git updates visible without copying individual scripts into PATH.
    run([*pip, "install", "--no-deps", "--editable", str(root)])
    # A new interpreter verifies disk state, independent of notebook imports.
    run([sys.executable, "-c", """
import importlib.metadata as metadata
import importlib.util
import cv2, numpy, torch, torchvision, ultralytics, scenedetect, lap, reframe
if importlib.util.find_spec('mediapipe') is not None:
    raise RuntimeError('mediapipe is still importable; inspect the reported Python environment')
for name in ('ultralytics', 'scenedetect', 'lap', 'numpy', 'opencv-python', 'torch', 'torchvision'):
    print(name, metadata.version(name), flush=True)
print('Python:', __import__('sys').executable, flush=True)
print('Reframe:', reframe.__version__, reframe.__file__, flush=True)
print('CUDA available:', torch.cuda.is_available(), 'CUDA build:', torch.version.cuda, flush=True)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
"""])
    run([sys.executable, "-m", "reframe", "--help"])
    print("Installed the reframe package and auto_reframe.py command. "
          "Keep this checkout for the editable installation.", flush=True)
    print("Restart the Colab session before running the updated pipeline if packages "
          "were already imported in notebook cells. Do not rerun the old installation cell.", flush=True)


if __name__ == "__main__":
    main()
