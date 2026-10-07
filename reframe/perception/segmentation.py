from __future__ import annotations

import logging
import math
from typing import Any
import cv2
import numpy as np
from ultralytics import YOLO
from reframe.config import AppConfig
from reframe.contracts import TrackObservation
from reframe.runtime import torch, verify_yolo_device

def mask_statistics(masks, indices: list[int], frame_shape: tuple[int, int]) -> dict:
    """Reduce only selected low-resolution masks; transfer four numbers per mask."""
    if masks is None or not indices:
        return {}
    height, width = frame_shape
    mh, mw = masks.shape[-2:]
    gain = min(mh / height, mw / width)
    pad_w = (mw - round(width * gain)) / 2
    pad_h = (mh - round(height * gain)) / 2
    left, top = round(pad_w - 0.1), round(pad_h - 0.1)
    right, bottom = mw - round(pad_w + 0.1), mh - round(pad_h + 0.1)
    if right <= left or bottom <= top:
        return {}
    sx, sy = width / (right - left), height / (bottom - top)
    valid = [i for i in indices if i < len(masks)]
    output = {}
    if torch is not None and torch.is_tensor(masks):
        # Optimization: Pre-allocate 1D coordinate tensors once outside the batch loop
        # to prevent redundant device allocations on every 16-mask chunk.
        xs = torch.arange(right - left, device=masks.device, dtype=torch.float32)
        ys = torch.arange(bottom - top, device=masks.device, dtype=torch.float32)
        for start in range(0, len(valid), 16):
            ids = valid[start : start + 16]
            binary = masks[ids, top:bottom, left:right] > 0.5
            rows = binary.sum(dim=2, dtype=torch.float32)
            cols = binary.sum(dim=1, dtype=torch.float32)
            count = rows.sum(dim=1)
            denom = count.clamp_min(1)
            cx = (cols * xs).sum(dim=1) / denom
            cy = (rows * ys).sum(dim=1) / denom
            ytop = (rows > 0).to(torch.int32).argmax(dim=1)
            compact = (
                torch.stack(
                    (
                        count * sx * sy,
                        (cx + 0.5) * sx - 0.5,
                        (cy + 0.5) * sy - 0.5,
                        ytop * sy,
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            output.update(
                {i: tuple(map(float, v)) for i, v in zip(ids, compact) if v[0] > 0}
            )
    else:
        for i in valid:
            small = masks[i, top:bottom, left:right]
            if torch is not None and torch.is_tensor(small):
                small = small.detach().numpy()
            binary = np.asarray(small > 0.5, dtype=np.uint8)
            moments = cv2.moments(binary, binaryImage=True)
            count = moments["m00"]
            if count:
                output[i] = (
                    count * sx * sy,
                    (moments["m10"] / count + 0.5) * sx - 0.5,
                    (moments["m01"] / count + 0.5) * sy - 0.5,
                    float(binary.any(axis=1).argmax()) * sy,
                )
    return output

def parse_tracks(result: Any, frame_shape: tuple[int, int], allowed_class_ids: list[int]) -> tuple[TrackObservation, ...]:
    h, w = frame_shape
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    raw = boxes.data
    raw = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)

    parsed_rows = []
    for i, row in enumerate(raw):
        try:
            cls_id = int(row[-1])
            if cls_id not in allowed_class_ids:
                continue
            conf = float(row[-2])
            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(row[4]) if len(row) == 7 else None

            if not all(math.isfinite(v) for v in (conf, x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                logging.warning("Ignoring invalid detection box at row=%s", i)
                continue

            parsed_rows.append((i, cls_id, conf, x1, y1, x2, y2, track_id))
        except Exception:
            logging.debug("Row parsing failed", exc_info=True)
            continue

    if not parsed_rows:
        return ()

    indices = [item[0] for item in parsed_rows]
    mask_data = getattr(getattr(result, "masks", None), "data", None)
    statistics = mask_statistics(mask_data, indices, (h, w))

    return tuple(TrackObservation(i, cls_id, conf, (x1, y1, x2, y2), track_id, statistics.get(i))
                 for i, cls_id, conf, x1, y1, x2, y2, track_id in parsed_rows)


class SegmentationTracker:
    """Owns the persistent segmentation model and resets IDs at scene cuts."""
    def __init__(self, config: AppConfig, allowed_class_ids: list[int]):
        self.config, self.allowed_class_ids = config, allowed_class_ids
        self.model = YOLO(config.seg_model)
        if self.model.task != "segment":
            raise ValueError(f"--seg-model requires a segmentation model, got {self.model.task!r}")
        self.class_names = self.model.names
        self.actual_device = None

    def reset(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()

    def track(self, frame: np.ndarray) -> tuple[TrackObservation, ...]:
        args = self.config
        result = self.model.track(source=frame, persist=True, tracker=args.tracker,
                                 classes=self.allowed_class_ids, conf=args.conf,
                                 retina_masks=args.retina_masks, device=args.yolo_device,
                                 verbose=False)[0]
        if self.actual_device is None:
            self.actual_device = verify_yolo_device(self.model, args.yolo_device)
        return parse_tracks(result, frame.shape[:2], self.allowed_class_ids)

    def close(self) -> None:
        self.model = None
