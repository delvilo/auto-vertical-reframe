# Local MSDB integration validation

These results validate the local phase-2 changes, not a Colab T4 performance or
crop-quality claim. Phase 2 is published on the draft PR branch after user authorization.
No ViNet or SAM2 was added. See COLAB_VALIDATION.md for the separate phase-1 GPU evidence.

## Configuration delivered

- Default backend: `deepgazemsdb`; old MR implementation and CLI choice removed.
- Default prior: `mit1003`; `--saliency-center-bias uniform` is available.
- `dataset=None`, original input frames, RGB 0–255, FP32 by default.
- Reference: 24-inch 9:16 portrait display at 60 cm; 1080×1920 output.
- Derived output PPD: 37.84346383587166 at the display centre.
- For a 1920×1080 input with 1080-pixel crop height: input PPD 21.28694840767781.
- For the supplied 3840×2160 source at 2160-pixel crop height: input PPD 42.57389681535562.
- Current pre-composition crop height controls the conversion; an explicit input
  PPD override is also available. The existing interval/EMA scheduler is retained.
- MSDB code/weight licence remains unconfirmed, as accepted for local integration.

## Checks completed

| Check | Result |
| --- | --- |
| Unit/regression suite (`unittest discover -s tests -v`) | 66 passed |
| Asset/device/fallback tests (`unittest test_security test_security_reframe -v`) | 5 passed |
| AST checkpoint loading check (`test_torch_load_weights_only`) | Passed |
| `pyflakes` on package, installer and new tests | Passed |
| `git diff --check` | Passed |
| Editable package installation and installed CLI `--help` | Passed |
| Official MIT1003 asset | Downloaded and validated: 1024×1024 finite log density |
| Official MSDB head | 25 entries accepted against pinned upstream architecture |
| Official CLIP RN50x64 and DINOv2 ViT-B/14 | Downloaded and loaded on CPU |
| Actual upstream MSDB forward | Passed, `deepgazemsdb/predicted`, zero fallback |

The tensor contract test uses a complete 3840×2160 BGR image with controlled model
outputs and verifies no input resizing, RGB order/range, prior normalization and
input PPD. Other tests cover uniform prior without template download, zoom/output
resolution conversion, invalid output, explicit CUDA failure, visible auto fallback,
and restoration of the scoped upstream constructor hooks after an exception.
Existing video regression tests use real OpenCV/FFmpeg with controlled neural outputs.

## Actual CPU model smoke test

- Environment: Python 3.12.14, PyTorch 2.8.0+cpu, torchvision 0.23.0+cpu,
  OpenCV 4.14.0.94, NumPy 2.5.3, DeepGaze 1.2.1 and the pinned OpenAI CLIP source.
- CPU PyTorch threads: 4. No CUDA device was available.
- Original synthetic image: 640×384, NumPy RNG seed 17, uint8 RGB-converted input.
- MIT1003 prior, `dataset=None`, explicit `pixel_per_dva=35` override, FP32.
- The 24-inch/60-cm derived PPD was **not** used in this smoke test.
- Initial loading: 79.970 seconds, including first downloads.
- First inference: 24.865 seconds, including input/output transfers and map conversion.
- Result: float32 map, shape `(384, 640)`, minimum 0.00001152185, maximum 1.0;
  `frames_backend=1`, `frames_fallback=0`.

These single-frame CPU timings are a functional smoke record. They cannot be
extrapolated to T4 video throughput. Upstream's xFormers-unavailable warnings were
left visible; no xFormers package or attention optimization was introduced.

## Remaining Colab validation

Use the MSDB cell in [README.md](README.md) with the complete local checkout.
The supplied phase-1 clip is 3840×2160 at 60 fps. Record the actual source-native input, model/device, zero fallback,
NVENC encoder, inference and total processing times, and CUDA allocator peaks.
The current model's internal scales may exceed T4 memory, especially at larger zoom;
the program must report OOM instead of silently resizing or replacing MSDB.
Full-source GPU inference, viewing-calibration quality, crop quality and speed
comparisons have not been measured here. Speed optimization remains the next step.
