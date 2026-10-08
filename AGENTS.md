# Development baseline

The user selected **Python 3.13+** as the compatibility baseline on 2026-10-08.

- Keep `requires-python` at `>=3.13` and use Python 3.13 or newer for development,
  package installation and verification.
- Compatibility with Python 3.12 and older is not required. Do not constrain new
  implementation choices solely to support those versions.
- Check Python package and accelerator compatibility against the actual Colab
  interpreter, installed PyTorch/torchvision builds and GPU. A newer Python
  version does not imply that every third-party CUDA wheel is available.
- Record the versions actually tested; do not claim that every future Python
  release has already been verified.

Current saliency baseline: DeepGaze MR with the phase-1 modular backend interface.
Preserve terminal stdout/stderr, native tracebacks and explicit backend telemetry.
