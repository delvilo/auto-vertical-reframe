# DeepGaze MR validation and cascade follow-up

Date: 2026-10-08. Package version: 0.2.1.

## Current optimization scope

The current CLI automatically selects pose-only composition, lightweight
handcrafted saliency with motion cues, or DeepGaze MR when the cheaper evidence
remains insufficient. `--saliency-model` is removed. Reliable body keypoints
are sufficient without requiring visible facial landmarks; pose remapping does
not renew the original inference time. Stable subjects receive priority pose
updates, with broader analysis for discovery, overlap and lost tracks.

`--precrop middle|left|right` analyzes half the source width at full height;
omitting it analyzes the whole frame. Inference coordinates are mapped back
to the source, while output is cropped from the full-resolution original.
The configured region never expands automatically to search for lost subjects.
Post-restore is disabled by default and all current recommended commands omit it;
the explicit `--post-restore` opt-in remains available.

DeepGaze observation keeps 16 low-resolution BGR uint8 frames on CPU without
loading the model. L3 requests load only after a complete temporal window exists,
then upload the missing frames. The model stays loaded across tier changes and
scene resets; each new scene still requires 16 fresh frames. Warmup and failures
use visible handcrafted fallback. A pose-only clip may have zero model loads and
zero neural calls. Existing trust configuration and weight-loading safeguards
remain in effect.

This change preserves the existing input scaling and saliency AMP defaults;
it does not add a fixed pixel budget, YOLO FP16, detection skipping or TensorRT.
Precropping can change aspect ratio and increase the resized DeepGaze pixel count.
No T4 speedup or visual-quality equivalence is claimed without new measurement.

Validation should cover the three half-width placements (including odd widths),
source-coordinate composition, pose freshness and stability, transitions and
cache expiry, lazy initialization, MR temporal continuity, scene resets,
resource cleanup, removed CLI options and default-disabled restoration. Colab
runs should report actual model calls and tier reasons alongside overall/stage
timing; compare stable people, overlap/occlusion, cuts and non-person footage.
Keep cold model loading separate from warmed processing time.

## Cascade verification in this workspace

Verified with Python 3.13.5, torch 2.11.0+cpu, torchvision 0.26.0+cpu,
NumPy 2.5.2, OpenCV 4.14.0.94, Ultralytics 8.4.173 and FFmpeg 7.1.5.
The editable package installs successfully; `pip check` and `git diff --check`
pass. The complete suite passes **105 tests and 29 subtests**, including pose
priority, all precrop coordinates, map provenance, tier transitions, adjacent-frame
cheap motion after a pose-only gap, lazy temporal upload, actual FFmpeg output
and resource cleanup. Tests use no downloaded neural weights.

Separate functional checks used the official YOLO26n segmentation and pose
weights with a generated 640×360, 12 fps clip of the Ultralytics bus image and
an AAC audio track. Full-frame, left, middle and right inference each produced
12-frame 180×320 H.264/AAC output, and all four outputs decoded without error.
This stationary fixture checks integration, not visual quality on moving footage.

| Inference region | Pose-only frames | Handcrafted frames | DeepGaze forwards | Model loaded |
| --- | ---: | ---: | ---: | --- |
| full | 9 | 3 | 0 | false |
| left | 9 | 3 | 0 | false |
| middle | 9 | 3 | 0 | false |
| right | 0 | 12 | 0 | false |

A longer 24-frame central-half check also decoded successfully: 21 pose-only
frames, 3 handcrafted frames, 15 inferred pose ROIs and 69 skipped competitor
ROIs. The temporal ring reached 16 frames while DeepGaze remained unloaded with
zero forwards. Post-restore was false in every render. ROI counts are not frame
counts, and a short fixture is not a speed benchmark.

A separate real-weight CPU cascade check used the official DeepGaze MR source
at `7beab05d96a36eb7b343995a10d5e91d46e52ddf`. The first 15 frames left the model
unloaded; frame 16 triggered exactly one real forward and returned a finite
128×64 saliency map, `model_loaded=true`, `frames_backend=1`, and no inference
fallback. The initial Hub fetch was blocked at the GitHub API; preparing the
official repository in the supported Torch Hub cache allowed retry. Its CUDA
checkpoint then used the existing CPU-safe local loader (`map_location="cpu"`,
`weights_only=True`) after the upstream loader warning. TLS/checksum and loader
trust safeguards were retained; warnings remained visible.

Functional artifacts are outside the checkout under
`/workspace/reframe-runtime/validation/cascade-*.log` and matching MP4 files.
Current summary fields include `elapsed_seconds`, `processing_fps`, and
`stage_wall_seconds`; stage values are host wall times, not isolated CUDA kernel
benchmarks. Tier counts, `saliency_actual_forward_calls`, `saliency_cache_hits`,
`saliency_cache_age_seconds` and pose ROI counts explain avoided work.

The new cascade has **not** been benchmarked or visually reviewed on Colab T4.
Use the current README command on the same source clip before reporting a GPU
speedup, memory saving or quality equivalence.

## Earlier rollback evidence

The results below record the earlier rollback baseline, not a rerun of the
new cascade.

The user withdrew the DeepGaze MSDB model update. This change restores the
phase-1 MR backend and its temporal input while preserving the modular pipeline,
YOLO26n-pose and terminal diagnostics. MR was restored as the default saliency backend for that baseline.
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

## Historical validation scope

The original rollback verification machine had no CUDA GPU. Subsequent uploaded
Colab logs did establish real-weight MR, YOLO26n-pose and HEVC NVENC success on
T4; see [the recorded 90-frame result](docs/colab-t4-results-2026-10-08.md).
Those logs used fixed-interval DeepGaze and post-restore. They do not validate
the subsequent automatic cascade, lazy loading, priority pose or precrop.
README contains current diagnostic and 90-frame cascade test cells; `tee` keeps
stdout/stderr visible and `pipefail` preserves a failed process exit status.

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
`predict` feature-reuse optimization. The new cascade reduces how often this forward call is required; reuse of
upstream per-frame VGG features remains separate work. The original Hub call follows the upstream default branch.
