"""Load only the published DeepGaze MR checkpoint and center bias.

No Hub Python code or ImageNet classifier weights are downloaded or executed.
Existing Hub data files are reused only after their full SHA256 is verified.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import tempfile
from typing import Any


UPSTREAM_COMMIT = "7beab05d96a36eb7b343995a10d5e91d46e52ddf"
ARTIFACT_SHA256 = {
    "deepgazemr-ledov.pt": "db0e3c4738664a6e8026229b8d1a74edda9ea1d571432273018a13c88a2c8ec6",
    "center-bias-ledov.pt": "cd6519218d45a7a18fc15e8dc9d09448fee2153264ecd1e7317d877274c5ce9d",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resolve_artifact(name: str, hub_dir: Path, torch: Any) -> Path:
    expected = ARTIFACT_SHA256[name]
    target = hub_dir / "checkpoints" / f"deepgazemr-{UPSTREAM_COMMIT}" / name
    candidates = [target, *sorted(hub_dir.glob(f"mtangemann_deepgazemr_*/data/{name}"))]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if _sha256(candidate) == expected:
            return candidate
        logging.warning("Ignoring DeepGaze artifact with unexpected SHA256: %s", candidate)

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/mtangemann/deepgazemr/{UPSTREAM_COMMIT}/data/{name}"
    logging.info("Downloading pinned DeepGaze artifact: %s", url)
    # Independent temporary files keep interrupted/concurrent downloads from
    # exposing partially written checkpoints to another process.
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.hub.download_url_to_file(url, str(temporary), hash_prefix=expected, progress=True)
        if _sha256(temporary) != expected:
            raise ValueError(f"DeepGaze artifact SHA256 mismatch: {name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _load_checkpoint_model(checkpoint_path: Path, bias_path: Path):
    import torch
    from reframe.saliency.backends.deepgaze_model import DeepGazeMR

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("DeepGaze checkpoint has no model_state_dict")
    bias = torch.load(bias_path, map_location="cpu", weights_only=True)
    if (not torch.is_tensor(bias) or bias.ndim != 2
            or not bool(torch.isfinite(bias).all()) or not bool((bias > 0).all())):
        raise ValueError("DeepGaze center bias must be a finite positive 2D tensor")
    model = DeepGazeMR(center_bias=bias)
    # The feature skeleton uses meta tensors; assign installs the actual
    # published tensors without allocating/initializing an unused VGG copy.
    # Missing keys, extra keys and mismatched shapes must remain fatal.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
    return model.eval()


def load_deepgaze_mr() -> tuple[Any, dict[str, Any]]:
    """Return the CPU model plus reproducible artifact provenance, on demand."""
    import torch

    hub_dir = Path(torch.hub.get_dir())
    checkpoint = _resolve_artifact("deepgazemr-ledov.pt", hub_dir, torch)
    bias = _resolve_artifact("center-bias-ledov.pt", hub_dir, torch)
    model = _load_checkpoint_model(checkpoint, bias)
    provenance = {
        "loader": "checkpoint_features_only",
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": ARTIFACT_SHA256["deepgazemr-ledov.pt"],
        "center_bias_path": str(bias),
        "center_bias_sha256": ARTIFACT_SHA256["center-bias-ledov.pt"],
        "imagenet_vgg_downloaded": False,
    }
    logging.info("DeepGaze direct checkpoint loader: %s", provenance)
    return model, provenance
