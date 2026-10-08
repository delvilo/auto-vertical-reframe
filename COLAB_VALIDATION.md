# Phase-1 Colab log review

Reviewed the user-supplied `reframe-modular-test.log` before publishing phase 2.
This records what the log establishes; output video quality was not inspected.

## Version and environment

- Package: `auto-vertical-reframe` 0.2.0, phase-1 branch commit
  `db1c73e427f85c7064a3ab7a697a20cdc7e847a5`.
- All 26 module hashes in the log match the phase-1 source files.
- Combined module fingerprint:
  `c4cdb83eaa952fd08b3691a3d49f0cf1e322153de526da937f3248d23927d898`.
- Python 3.13.15; PyTorch 2.11.0+cu130; CUDA build 13.0;
  CUDA available; Tesla T4, compute capability 7.5.
- Source: `vv110.mp4`, **3840×2160, 60 fps**, about 172.01 seconds / 10,320 frames.
  A directory name containing `1080p` does not describe the actual source size.

## Observed result

| Item | Evidence in the log |
| --- | --- |
| Processed frames | 90, corresponding to 1.5 seconds of source video |
| Segmentation / pose devices | Both actually on `cuda:0` |
| Subject / pose availability | 90 / 90 frames |
| Head-cue availability | 48 frames; cue availability is not an accuracy score |
| Pose inference / matches | 19 ROIs inferred, 19 matched; 193 cache hits |
| Subject switches | 0 |
| Scene resets | 1, including initial scene setup |
| Saliency | Requested and active `handcrafted`; 30 refreshes, 60 propagated frames |
| Last saliency result | Frame 90, inferred at frame 88 / 1.45 s; map 384×216 |
| Saliency fallback | 0; `model_loaded=false` is normal for handcrafted |
| Output | 1080×1920, 60 fps, HEVC via `hevc_nvenc`; audio transcoded to AAC |
| FFmpeg completion | 90 frames, exit code 0 |
| Errors | No logged ERROR, WARNING or traceback; original MediaPipe messages absent |

The CUDA detection/pose path and NVENC encoding therefore passed this short run.
The last encoded timestamp of 1.48 seconds is consistent with a 90-frame, 60-fps
sequence; it does not indicate missing frames.

## Limits and phase-2 test

- This run used handcrafted saliency. It does **not** validate DeepGaze MSDB,
  model downloads, viewing-scale conversion or full-frame neural inference.
- The final FFmpeg `8.0 fps` is its stream-processing rate, not a separately
  measured end-to-end benchmark. Phase 2 adds `wall_seconds`, `processing_fps`,
  model loading/inference timings and process CUDA allocator peaks.
- A 90-frame run cannot establish full-video stability or cross-scene behavior.
  Framing quality and audio synchronization still require viewing the output.
- Phase-2 Colab instructions in [README.md](README.md) first process 3 frames,
  then 90 frames using MSDB, MIT1003, a 24-inch portrait display at 60 cm, full
  original frames and the existing saliency interval of 3. Both runs preserve
  terminal output and their complete logs. The second run starts only if the
  first succeeds.
- A T4 MSDB result must show `active_backend=deepgazemsdb`, `device=cuda:0`,
  `input_size=[3840,2160]`, zero fallback and the actual `hevc_nvenc` encoder.
  Successful process completion and output inspection are also required.
  CUDA OOM must remain visible; no automatic shrinking or backend substitution
  is enabled by these test commands.

Local phase-2 checks and the separate real-weight CPU smoke test are recorded in
[MSDB_VALIDATION.md](MSDB_VALIDATION.md). Model licensing remains unconfirmed as
recorded in [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).
