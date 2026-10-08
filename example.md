# Command examples (Python 3.13+)

All examples use automatic three-tier composition: reliable pose, lightweight
saliency, then lazy DeepGaze MR when needed. `--saliency-model` has been removed.
FFmpeg denoising and sharpening are disabled by default; add `--post-restore`
only when you intentionally want those additional filters.

## Colab T4: central-half inference and subject lock

```bash
python3 -m reframe input.mp4 output_vertical.mp4 \
    --preset talking_head \
    --device 0 \
    --precrop middle \
    --lock-first-subject \
    --saliency-device cuda \
    --saliency-trust-repo \
    --no-saliency-amp \
    --video-encoder hevc_nvenc
```

`middle` analyzes x=25%–75%, at full height. Use `left` for x=0%–50%,
`right` for x=50%–100%, or omit `--precrop` for the whole frame. Detection,
pose and saliency stay inside that region; output crops the full-resolution
source and may extend outside the inference region. The program does not search
outside the chosen region when a subject leaves it. Half-width inference does
not guarantee half the compute because model input resizing still applies.

## Fixed zoom and initial subject

```bash
python3 -m reframe input.mp4 output_fixed.mp4 \
    --preset talking_head \
    --lock-first-subject \
    --fixed-zoom 1.10 \
    --dead-zone 0.08 \
    --pan-time 0.55 \
    --saliency-trust-repo
```

## Two-person podcast

```bash
python3 -m reframe podcast.mp4 output_podcast.mp4 \
    --preset talking_head \
    --two-person-framing \
    --two-person-threshold 0.75 \
    --switch-score-threshold 1.25 \
    --min-subject-hold-frames 18 \
    --saliency-trust-repo
```

## Sports and action

```bash
python3 -m reframe skate.mp4 output_sports.mp4 \
    --preset sports \
    --classes person bicycle motorcycle \
    --motion-response 0.18 \
    --motion-damping 0.75 \
    --max-step-x 16.0 \
    --saliency-trust-repo
```

## Debug preview and bounded validation

```bash
python3 -u -m reframe footage.mp4 output_test.mp4 \
    --preset talking_head \
    --saliency-trust-repo \
    --save-debug-preview \
    --debug-path output_debug_hud.mp4 \
    --max-frames 90 \
    --native-debug \
    --log-level DEBUG \
    --ffmpeg-log-level info
```

Compare stable people, crossing/occlusion, scene cuts and non-person subjects.
Check model call counts and tier reasons as well as runtime and crop quality.
A stable clip can legitimately finish without loading DeepGaze; use an ambiguous
clip longer than 16 frames to exercise its neural path. Actual T4 speedup must
be measured on the same source and settings, with cold-start time separated.
