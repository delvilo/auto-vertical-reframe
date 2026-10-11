"""YOLO tensor input with original-image geometry and native tracking callbacks.

Ultralytics normally treats a BCHW source's letterboxed dimensions as the original
image and downloads that image during postprocessing. These predictors preserve
the source ROI dimensions without copying source pixels to the CPU. Results carry
geometry-only images, so this adapter deliberately supports ByteTrack only and
disables image visualization/export. It is exercised against Ultralytics 8.4.173.
"""
from __future__ import annotations

import numpy as np
import torch
from ultralytics.data.loaders import LoadTensor, SourceTypes
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_imgsz, check_yaml

from reframe.frame_pixels import TensorFrame


def geometry_image(shape: tuple[int, int, int]) -> np.ndarray:
    """A three-byte, read-only broadcast view; no source image allocation."""
    return np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), shape)


class _LetterboxedTensorLoader(LoadTensor):
    """Our own uint8-to-float letterbox guarantees [0, 1], without a GPU scan.

    Native LoadTensor tests ``im.max()`` in Python. Repeating that check on a
    tensor we just constructed adds a CUDA synchronization for each prediction.
    This private loader only receives TensorFrame.letterbox output; arbitrary
    user tensors still use Ultralytics' normal validation elsewhere.
    """

    @staticmethod
    def _single_check(im: torch.Tensor, stride: int = 32) -> torch.Tensor:
        if (im.ndim != 4 or not all(im.shape) or im.shape[1] != 3
                or im.shape[2] % stride or im.shape[3] % stride
                or im.dtype != torch.float32):
            raise ValueError("Internal YOLO letterbox must be float32 RGB NCHW with stride-aligned dimensions")
        return im

    def __init__(self, im0: torch.Tensor):
        super().__init__(im0)
        self.source_type = SourceTypes(tensor=True)


class _TensorOriginalGeometry:
    def setup_source(self, source, stride: int | None = None):
        frames = [source] if isinstance(source, TensorFrame) else source
        if not isinstance(frames, (list, tuple)) or not frames or not all(
            isinstance(frame, TensorFrame) for frame in frames
        ):
            raise TypeError("Tensor YOLO predictor requires TensorFrame inputs")
        if any(getattr(self.args, key, False) for key in
               ("save", "save_txt", "save_crop", "show", "visualize")):
            raise ValueError("Tensor YOLO geometry results do not support image visualization/export")
        if self.args.mode == "track":
            if len(frames) != 1:
                raise ValueError("Tensor ByteTrack requires exactly one source frame per update")
            tracker = str(self.args.tracker)
            if getattr(self, "_validated_tensor_tracker", None) != tracker:
                settings = YAML.load(check_yaml(tracker))
                if settings.get("tracker_type") != "bytetrack":
                    raise ValueError("Tensor frame processing supports ByteTrack only; "
                                     "BoT-SORT/GMC/ReID require original CPU pixels")
                self._validated_tensor_tracker = tracker
            # Reject externally replaced trackers before postprocessing invokes
            # native callbacks with geometry-only images.
            from ultralytics.trackers.byte_tracker import BYTETracker
            if any(type(tracker) is not BYTETracker for tracker in getattr(self, "trackers", [])):
                raise ValueError("Tensor frame processing requires unmodified native ByteTrack")
        if getattr(self.model, "format", None) != "pt":
            raise ValueError("Tensor frame processing currently requires a PyTorch .pt YOLO model")
        stride = int(stride or self.model.stride)
        size = check_imgsz(self.args.imgsz, stride=stride, min_dim=2)
        same_shapes = len({tuple(frame.shape) for frame in frames}) == 1
        auto = bool(same_shapes and self.args.rect)
        inputs = [frame.letterbox(size=tuple(size), auto=auto, stride=stride)[0]
                  for frame in frames]
        source_tensor = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
        self._tensor_originals = [geometry_image(tuple(frame.shape)) for frame in frames]
        # Native load_inference_source does not accept an existing LoadTensor
        # instance. Set the same predictor fields directly for this narrow
        # in-memory path; stream_inference and tracking callbacks stay native.
        self.imgsz = size
        self.dataset = _LetterboxedTensorLoader(source_tensor)
        self.source_type = self.dataset.source_type
        self.vid_writer = {}

    def postprocess(self, preds, img, orig_imgs):
        # A list bypasses convert_torch2numpy_batch in DetectionPredictor. Its
        # standard construct_result scales boxes, masks, and keypoints back to
        # the ORIGINAL inference region before the tracking callback runs.
        return super().postprocess(preds, img, self._tensor_originals)


class TensorSegmentationPredictor(_TensorOriginalGeometry, SegmentationPredictor):
    pass


class TensorPosePredictor(_TensorOriginalGeometry, PosePredictor):
    pass


def predictor_options(model, predictor_type) -> dict:
    """Never silently reuse a NumPy predictor, which loses tensor ROI geometry."""
    current = getattr(model, "predictor", None)
    if current is not None and not isinstance(current, predictor_type):
        raise ValueError("Cannot switch an initialized YOLO model between NumPy and tensor frames; "
                         "start a new processing run")
    return {"predictor": predictor_type, "save": False, "save_txt": False,
            "save_crop": False, "show": False, "visualize": False}
