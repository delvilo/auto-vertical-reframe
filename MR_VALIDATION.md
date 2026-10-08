# DeepGaze MR rollback validation

Date: 2026-10-08. Package version: 0.2.1.

The user withdrew the DeepGaze MSDB model update. This change restores the
phase-1 MR backend and its temporal input while preserving the modular pipeline,
YOLO26n-pose and terminal diagnostics. MR is now the default saliency backend.
MSDB code, assets, viewing parameters and Python dependencies are removed.
The user's subsequent README edit removing the demo section is preserved.

## Python baseline

Python **3.13+** is the user-selected development and installation baseline.
`pyproject.toml`, `install_colab.py`, `requirements.txt`, README and `AGENTS.md`
record that decision. Future Python and CUDA wheel combinations still require
verification; the baseline is not a claim that every future release was tested.

The local verification environment used Python 3.13.16, PyTorch 2.8.0+cpu,
torchvision 0.23.0+cpu, NumPy 2.5.2, OpenCV 4.14.0.94, Ultralytics 8.4.173,
PySceneDetect 0.7.1 and lap 0.5.13. Editable installation succeeded and the
installed package reports version 0.2.1 and `Requires-Python: >=3.13`.

## Local evidence

- 61 tests under `tests/` passed, including backend contracts, temporal ingestion,
  scene resets, resource cleanup, diagnostic output and real FFmpeg smoke tests.
- Five existing security tests and the `weights_only=True` AST check passed.
- Pyflakes and whitespace checks passed. CLI parsing accepts MR and max-side 768.
- An additional real-weight CPU smoke test loaded the official MR model, observed
  16 RGB frames of size 96×64, and produced a finite float32 map of size 96×64.
  Telemetry reported `model_loaded=true`, `active_backend=deepgazemr`,
  `observed_frames=16`, `frames_backend=1` and `frames_fallback=0`.

The smoke test used `TORCH_HOME` in a separate scratch cache, `trust_repo=True`
and AMP disabled. The official Hub loader initially failed on CPU because its
checkpoint contains CUDA storage. The restored local-cache loader then loaded
the same weights with `map_location="cpu", weights_only=True` and successfully
ran MR. The original warning and traceback remained visible; handcrafted output
was not accepted as a successful MR smoke test. Torchvision also emitted the
upstream `pretrained` deprecation warnings.

## Colab validation still required

There is no CUDA GPU in the local verification environment. The earlier
successful Colab phase-1 log used handcrafted saliency and does not establish
real-weight MR success on T4. README contains the branch synchronization,
installation and 90-frame MR test cells. Both stdout and stderr remain visible
and are saved with `tee`; `pipefail` preserves a failed process exit status.

With `saliency_interval=3`, MR window warmup normally contributes five refreshes
to the legacy `saliency_frames_fallback` counter. Check actual model loading,
CUDA device and positive `saliency_frames_backend`, rather than requiring zero
fallback refreshes. Scene cuts restart the 16-frame window.

The default maximum input side remains 384 (384×216 for large 16:9 sources).
Max-side 768 preserves more detail at four times the input pixel count, but no
accuracy improvement or T4 latency/VRAM result has been measured. Compare the
same representative video segments with other settings held constant before
changing the default. The final video and YOLO input resolutions are unaffected
by this MR setting.

## Upstream reference

[DeepGaze MR](https://github.com/mtangemann/deepgazemr), source inspected at
`7beab05d96a36eb7b343995a10d5e91d46e52ddf`:
the official `forward` input is a 16-frame clip. VGG19 features are averaged
over time; optical flow is not an additional model input. Our restored adapter
uses `forward` on each scheduled window, so it does not yet use the upstream
`predict` feature-reuse optimization. That remains part of the subsequent
performance work. The original Hub call follows the upstream default branch.
