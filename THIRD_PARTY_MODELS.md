# Third-party model provenance and licence status

DeepGaze MSDB code and pretrained-weight licensing is **UNCONFIRMED**.
The user approved proceeding with local integration with this status recorded.
This is not a commercial-use permission or a grant of rights to redistribute the
upstream code, weights, or centre-bias data. This project's MIT licence does not
cover third-party models. No third-party source or weights are bundled here.

As checked for this integration, the [DeepGaze repository](https://github.com/matthias-k/DeepGaze)
has no repository licence file and its MIT entries in
[setup.py](https://github.com/matthias-k/DeepGaze/blob/c7db17e2d1d7ea6468ffdee2cfaddf141095dcff/setup.py)
are commented out. [Issue #15](https://github.com/matthias-k/DeepGaze/issues/15)
asks about commercial licensing without an answer. The author's reply in
[issue #13](https://github.com/matthias-k/DeepGaze/issues/13) grants CC-BY to a
specific illustration, not to the model implementation or pretrained weights.

| Component | Source/version used |
| --- | --- |
| DeepGaze MSDB implementation | `matthias-k/DeepGaze@c7db17e2d1d7ea6468ffdee2cfaddf141095dcff` (package 1.2.1) |
| MSDB saliency head | [v1.2.0/deepgazemsdb.pth](https://github.com/matthias-k/DeepGaze/releases/download/v1.2.0/deepgazemsdb.pth) |
| MIT1003 centre-bias template | [v1.0.0/centerbias_mit1003.npy](https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy) |
| OpenAI CLIP RN50x64 | `openai/CLIP@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`; official CLIP loader/checkpoint URL |
| DINOv2 ViT-B/14 | `facebookresearch/dinov2@6a6261546c3357f2c243a60cfafa6607f84efcb7`; official `dinov2_vitb14_pretrain.pth` |

CLIP, DINOv2, YOLO and other dependencies retain their own upstream terms. Review
those terms together with DeepGaze's unresolved status for the intended use.

Python source packages are installed from the pinned repositories in
`requirements.txt`. On first MSDB use, the adapter downloads official model assets
to the normal PyTorch/CLIP user caches. DINO's pinned Hub entry point executes
upstream code with `trust_repo=True`; this is limited to the fixed repository and
commit above, with no CLI option to execute an arbitrary Hub repository. The old
MR-specific `--saliency-trust-repo` flag has been removed.

DINO and MSDB checkpoints use `weights_only=True`; the complete MSDB head key set
is checked before accepting its intentionally backbone-free checkpoint. The NumPy
prior is read with `allow_pickle=False`. OpenAI's CLIP loader checks the published
checkpoint's SHA256. Package pins identify source revisions, not an independent
security audit or a licence determination.

ViNet and SAM2 are not integrated or installed. MediaPipe remains removed.
