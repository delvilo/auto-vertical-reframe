# DeepGaze MR validation and cascade follow-up

Updated: 2026-10-09 (Asia/Taipei). Package version: 0.2.1.

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
it does not add a fixed pixel budget, YOLO FP16 or TensorRT. Adaptive detector
skipping with sparse optical flow is now enabled as described below.
Precropping can change aspect ratio and increase the resized DeepGaze pixel count.
No T4 speedup or visual-quality equivalence is claimed without new measurement.

Validation should cover the three half-width placements (including odd widths),
source-coordinate composition, pose freshness and stability, transitions and
cache expiry, lazy initialization, MR temporal continuity, scene resets,
resource cleanup, removed CLI options and default-disabled restoration. Colab
runs should report actual model calls and tier reasons alongside overall/stage
timing; compare stable people, overlap/occlusion, cuts and non-person footage.
Keep cold model loading separate from warmed processing time.

## Lazy tracking thumbnails and timing collection (2026-10-09)

The four uploaded `reframe-seg-gap{1,3}-run{1,2}.log` files match all 29 Python
module hashes at `6476628`. On their common 600-frame, 4K/60 fps, middle-precrop
T4 input, the second (cached-weight) round showed a regression:

| Measurement | gap 1 | gap 3 (original eager thumbnails) |
| --- | ---: | ---: |
| Total pipeline time | 117.139 s | 130.457 s |
| Segmentation stage | 18.100 s | 32.111 s |
| YOLO calls / predicted frames | 600 / 0 | 587 / 13 |
| Sparse motion estimation | 0 | 0.063 s |
| Pose ROI inferences / MR forwards | 328 / 135 | 328 / 135 |

Both gap 3 runs made identical decisions. The extra 14.011 s in segmentation
cannot be attributed to sparse flow itself. The first gap 1 run downloaded the
Hub repository and VGG weights, so its 136.761 s is not a fair warm baseline.
All four runs completed 600 output frames at 60 fps with HEVC NVENC and AAC,
with equal subject/pose/tier counts. These logs do not establish visual equality.

The new default `--seg-thumbnail-mode lazy` runs the original metadata-only
refresh checks before building a thumbnail. The later image-change, confidence,
age and scheduled-update ordering is preserved, as are thresholds and the
320-pixel INTER_AREA resize. If required, frame N-1 is lazily reconstructed from
one retained read-only source reference, never from an older gray image. Each
frame is resized at most once. When a gray image exists, its raw reference is
released; reset and close release both. Retaining a half-width crop view can keep
its entire 4K BGR backing array alive (about 25 MB host RAM).

`--seg-thumbnail-mode eager` retains the original every-frame thumbnail behavior
with the same new instrumentation. It provides a same-version reference without
changing detector, pose or saliency policies. The metadata gate reasons covered
310 of the uploaded frames, but deferred images may later require backfilling;
that count is not a promise of 310 saved resizes.

Summary now includes exclusive host-wall buckets in `seg_timing_seconds`:
`thumbnail`, `scheduler`, `frame_change`, `model_track`, `parse_masks`, `flow`,
`predict` and `bookkeeping`. Their sum equals `seg_total_seconds`; the outer
pipeline segmentation stage additionally includes call-boundary overhead.
`scheduler` covers decisions/stability inside `track()`; existing post-pose
feedback remains within the saliency stage. `parse_masks` can absorb GPU-to-CPU
waiting, and `model_track` includes YOLO preprocessing/postprocessing and native
tracking. No new per-frame CUDA synchronization is introduced. Existing
`seg_flow_seconds` aliases the flow bucket and must not be added again.

Thumbnail builds, cache hits, prior-frame backfills, current-frame deferrals and
flow attempts are counted separately. Deferrals are not permanently saved
resizes. These counters and timings determine whether avoiding thumbnail work
actually pays off on T4; a new speedup has not yet been measured there.

Python 3.13.5 verification passes **164 tests and 84 subtests**. New tests cover
exact eager/lazy decisions and boxes across forced updates, scene and geometry
changes; adjacent KLT inputs after deferred runs; single builds per frame;
resource release; timing reconciliation; collector failure/exit-code handling;
warmup exclusion; and actual frame/FPS/audio validation logic.

`scripts/benchmark_colab.py` runs gap1-lazy, gap3-eager and gap3-lazy, one warmup
each plus two measured rounds by default (nine processes). Measured order is
reversed on alternate rounds. It keeps native logs, commands, environment and
ffprobe metadata, all Summary values, flattened CSV and per-case medians.
It rejects stale timing schemas, inconsistent timing totals, wrong frame/FPS
counts, missing audio and unexpected encoder fallback, while preserving failed
run evidence. Warmups are excluded from medians; independent model startup is
still included in each process. See README for runnable Colab cells.

A real-model CPU smoke run also completed all six invocations (three warmups
and one measured repeat per case) using the existing 12-frame, 12 fps fixture,
official YOLO26n segmentation/Pose weights and libx264. Every output retained
12 frames, 12 fps and audio; JSON/CSV/medians and logs were generated and validated.
Eager and lazy each built 12 thumbnails on this short stable fixture; it tests
collector integration, not the expected savings on forced-detection sequences.
Local evidence is under
`/workspace/reframe-runtime/validation/thumbnail-collector-smoke-20261009/`.

## Adaptive segmentation verification (2026-10-08)

Default `--seg-max-gap 3 --seg-max-age 0.1` starts with every-frame segmentation.
Matching real detections must remain stable for 0.1 video seconds to reach gap 2,
and 0.25 seconds to reach gap 3. Predictions cannot advance those measurement
timestamps. Each skipped frame uses a <=320-pixel thumbnail, forward/backward
sparse KLT and robust translation. Low feature counts, inconsistent flow, strong
image changes, overlap, crop boundaries, uncertain tracking and weak fresh primary
pose request a real detection. The current policy never exceeds gap 3, even if
the configured upper bound is larger. `--seg-max-gap 1` is the baseline.

The pinned native ByteTrack advances its Kalman prediction and frame counter once
per source frame, without calling an empty detection update on skipped frames.
Flow corrects the center without shrinking detector covariance or renewing its
measurement time. Lost/unconfirmed tracks and unsupported tracker classes use
every-frame detection. Masks retain short-lived translated moments; no new full
segmentation mask is fabricated. Pose, cascade and ranking reject stale or
unreliable predictions; precrop translates measurement geometry back to source
coordinates while preserving the original timestamp.

Python 3.13.5 regression verification passed **143 tests and 72 subtests** using
the CPU dependency versions listed below. New checks include actual OpenCV flow
and real ByteTrack association/Kalman state with synthetic detections, slow
translation, unchanged IDs and frame clocks, measurement age, scene/geometry
resets, lost tracks, unsupported trackers, invalid flow, and downstream provenance.

A separate end-to-end CPU comparison used official YOLO26n segmentation/Pose and
DeepGaze MR weights. The fixture is a stationary Ultralytics bus photograph,
resized to 240x320 and padded to a 640x360 video: 90 frames at 60 fps, with AAC
audio. Both runs used `--precrop middle --conf 0.45`, 180x320 libx264 output,
default saliency resolution/AMP and no post-restore. The 0.45 confidence threshold
excludes a weak partial person in this fixture; it is not a new application
default. Weights were already cached; separate process/model startup is included.

| Measurement | Every frame (gap 1) | Adaptive (gap 3) |
| --- | ---: | ---: |
| YOLO segmentation calls | 90 | 36 |
| Flow-predicted frames | 0 | 54 |
| Segmentation stage, including scheduling/flow | 8.498 s | 4.523 s |
| Sparse flow time (included above) | 0 | 0.296 s |
| Total pipeline time | 47.953 s | 41.793 s |
| Pose ROI inference calls | 10 | 10 |
| DeepGaze forward calls | 4 | 4 |
| Subject switches | 0 | 0 |
| Maximum prediction age | 0 | 0.0333 s |

Both outputs decode successfully to **90 frames, 60 fps, 1.500 seconds**.
Their decoded pixels are identical on this stationary fixture. Both retain AAC
with the same 0.000-second start and 1.493-second encoded duration (source audio
is 1.500 seconds; existing encoding/trimming is unchanged). DeepGaze ingested all
90 source frames in each run. CPU loading used the existing safe local fallback
after the upstream CUDA checkpoint warning, with four successful neural forwards.

The observed 60% reduction is in YOLO calls, not overall runtime. This single
short stationary fixture validates integration; it does not establish moving-video
quality or a T4 speedup. GPU validation should compare warmed 600-frame runs on
stable people, entrances, overlap/occlusion, cuts and non-person footage. README
provides commands and telemetry keys. Raw local artifacts are under
`/workspace/reframe-runtime/validation/adaptive-{gap1,gap3}.{log,mp4}` and
`adaptive-comparison.json`; they are not committed model/video assets.

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
