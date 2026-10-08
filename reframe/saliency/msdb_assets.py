"""Pinned upstream MSDB integration; no upstream source or weights are bundled.

DeepGaze's code/weight licence remains unconfirmed. See THIRD_PARTY_MODELS.md.
The scoped factory replacements keep backbone construction on CPU and make the
DINO revision and tensor-only checkpoint loading explicit. Model layers, feature
hooks, scale lists and forward computation remain upstream implementations.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import tempfile
from threading import RLock
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from reframe.runtime import torch

DEEPGAZE_REVISION = "c7db17e2d1d7ea6468ffdee2cfaddf141095dcff"
CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
DINO_REPOSITORY = "facebookresearch/dinov2:6a6261546c3357f2c243a60cfafa6607f84efcb7"
MSDB_WEIGHTS_URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.2.0/deepgazemsdb.pth"
DINO_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
CENTER_BIAS_URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
_CONSTRUCTION_LOCK = RLock()


@contextmanager
def _cpu_backbone_factories(upstream):
    # Upstream has no device/factory constructor arguments: CLIP otherwise picks
    # CUDA even when the caller requests CPU. Patch only its two constructor
    # references, never torch/clip globally, and restore even if loading fails.
    import clip
    from deepgaze_pytorch.features.normalizer import CLIP_Normalizer, Normalizer

    def clip_factory():
        logging.info("Loading OpenAI CLIP RN50x64 backbone on CPU")
        model, _ = clip.load("RN50x64", device="cpu", jit=False)
        visual = model.visual
        visual.attnpool = torch.nn.Sequential()
        return torch.nn.Sequential(CLIP_Normalizer(), visual)

    def dino_factory():
        logging.info("Loading DINOv2 backbone from pinned %s", DINO_REPOSITORY)
        model = torch.hub.load(DINO_REPOSITORY, "dinov2_vitb14", pretrained=False,
                               trust_repo=True, skip_validation=True)
        weights = torch.hub.load_state_dict_from_url(
            DINO_WEIGHTS_URL, map_location="cpu", weights_only=True, progress=True
        )
        model.load_state_dict(weights, strict=True)
        return torch.nn.Sequential(Normalizer(), model)

    with _CONSTRUCTION_LOCK:
        original = upstream.CLIPResNet50x64, upstream.DINOv2_ViTB14
        upstream.CLIPResNet50x64, upstream.DINOv2_ViTB14 = clip_factory, dino_factory
        try:
            yield
        finally:
            upstream.CLIPResNet50x64, upstream.DINOv2_ViTB14 = original


def load_msdb_model():
    from deepgaze_pytorch import deepgazemsdb as upstream
    with _cpu_backbone_factories(upstream):
        model = upstream.DeepGazeMSDB(pretrained=False)
    checkpoint = torch.hub.load_state_dict_from_url(
        MSDB_WEIGHTS_URL, map_location="cpu", weights_only=True, progress=True
    )
    # The published checkpoint deliberately omits the two pretrained backbones.
    # strict=False alone could silently accept an incomplete/random saliency head.
    expected = set(model.head_state_dict())
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected:
        actual = set(checkpoint) if isinstance(checkpoint, dict) else set()
        raise ValueError(f"Invalid MSDB head checkpoint: missing={sorted(expected - actual)} "
                         f"unexpected={sorted(actual - expected)}")
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("features.backbone.") for key in incompatible.missing_keys
    ):
        raise ValueError(f"MSDB checkpoint mismatch: {incompatible}")
    model.float()
    model.requires_grad_(False)
    model.eval()
    return model


def load_mit1003_template() -> np.ndarray:
    directory = Path(torch.hub.get_dir()) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "centerbias_mit1003.npy"
    if not destination.is_file():
        logging.info("Downloading MIT1003 center bias: %s", CENTER_BIAS_URL)
        with tempfile.TemporaryDirectory(prefix="centerbias-", dir=directory) as temporary:
            download = Path(temporary) / destination.name
            torch.hub.download_url_to_file(CENTER_BIAS_URL, str(download), progress=True)
            template = _read_template(download)
            download.replace(destination)
            return template
    logging.info("Loading MIT1003 center bias: %s", destination)
    return _read_template(destination)


def _read_template(path: Path) -> np.ndarray:
    template = np.load(path, allow_pickle=False)
    if (template.ndim != 2 or min(template.shape) < 2
            or not np.issubdtype(template.dtype, np.number) or not np.isfinite(template).all()):
        raise ValueError(f"Invalid MIT1003 center bias: {path}")
    return template.astype(np.float64)


def center_bias_log_density(shape: tuple[int, int], template: np.ndarray | None) -> np.ndarray:
    if template is None:
        density = np.zeros(shape, dtype=np.float64)
    else:
        density = zoom(template, (shape[0] / template.shape[0], shape[1] / template.shape[1]),
                       order=0, mode="nearest")
    density -= logsumexp(density)
    return density.astype(np.float32)
