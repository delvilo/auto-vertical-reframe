"""Shared CPU COCO evidence gates for scheduling and pose-only framing.

Face/hand visibility is independent of torso credibility. These gates describe
the last actual pose measurement, not evidence inferred by translating a box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TORSO = (5, 6, 11, 12)
LIMBS = (7, 8, 9, 10, 13, 14, 15, 16)


@dataclass(frozen=True)
class PoseReliability:
    valid_mask: np.ndarray
    torso_mask: np.ndarray
    torso_reliable: bool
    face_reliable: bool
    limb_reliable: bool
    reason: str


@dataclass(frozen=True)
class PoseMotion:
    torso_displacement: float
    limb_displacement: float
    torso_common: int
    limb_common: int


def assess_pose(keypoints, box, keypoint_conf: float = .35) -> PoseReliability:
    """Require a confident, noncollapsed torso; do not require a visible face.

    Three shoulders/hips suffice when both anatomical groups are represented.
    The .5 torso floor preserves the primary scheduler's confidence threshold;
    lowering a display/limb confidence option cannot authorize weak cache reuse.
    Coordinates are normalized separately by box width/height for scale safety.
    Side-on/back views are allowed: no assumption about left/right image order.
    """
    valid = np.zeros(17, dtype=bool)
    torso = valid.copy()

    def result(reliable=False, reason="invalid_pose"):
        return PoseReliability(valid, torso, reliable, bool(valid[:5].sum() >= 2),
                               bool(valid[list(LIMBS)].sum() >= 2), reason)

    points = np.asarray(keypoints) if keypoints is not None else np.empty((0, 3))
    bounds = np.asarray(box, dtype=float)
    if (points.shape != (17, 3) or bounds.shape != (4,)
            or not np.isfinite(bounds).all() or not np.isfinite(keypoint_conf)):
        return result()
    size = bounds[2:] - bounds[:2]
    if np.any(size <= 0):
        return result()
    valid = (np.isfinite(points).all(axis=1) & (points[:, 2] >= keypoint_conf)
             & (points[:, 0] >= bounds[0] - .12 * size[0])
             & (points[:, 0] <= bounds[2] + .12 * size[0])
             & (points[:, 1] >= bounds[1] - .12 * size[1])
             & (points[:, 1] <= bounds[3] + .12 * size[1]))
    torso[list(TORSO)] = valid[list(TORSO)] & (points[list(TORSO), 2] >= max(.5, keypoint_conf))
    if torso.sum() < 3:
        return result(reason="insufficient_torso")
    normalized = (points[:, :2] - bounds[:2]) / size
    shoulders = normalized[[5, 6]][torso[[5, 6]]]
    hips = normalized[[11, 12]][torso[[11, 12]]]
    separation = np.linalg.norm(shoulders.mean(axis=0) - hips.mean(axis=0))
    if not .10 <= separation <= 1.10:
        return result(reason="collapsed_or_implausible_torso")
    for group, minimum, maximum in ((shoulders, .035, 1.15), (hips, .025, 1.0)):
        if len(group) == 2 and not minimum <= np.linalg.norm(group[0] - group[1]) <= maximum:
            return result(reason="collapsed_or_implausible_torso")
    return result(True, "reliable_torso")


def pose_motion(previous_points, previous_box, points, box,
                keypoint_conf: float = .35) -> PoseMotion:
    """Compare actual pose shape; invalid wrists do not masquerade as motion.

    Confident large arm/leg motion is retained separately as a reason to refresh
    rather than getting diluted in the average across stationary torso points.
    """
    before = assess_pose(previous_points, previous_box, keypoint_conf)
    current = assess_pose(points, box, keypoint_conf)
    if not before.torso_reliable or not current.torso_reliable:
        return PoseMotion(float("inf"), 0.0, 0, 0)
    common_torso = before.torso_mask & current.torso_mask
    common_limb = before.valid_mask & current.valid_mask
    common_limb[list(set(range(17)) - set(LIMBS))] = False

    def relative(values, bounds):
        bounds = np.asarray(bounds, dtype=float)
        return (np.asarray(values)[:, :2] - bounds[:2]) / (bounds[2:] - bounds[:2])

    displacement = np.linalg.norm(relative(points, box) - relative(previous_points, previous_box), axis=1)
    torso_count, limb_count = int(common_torso.sum()), int(common_limb.sum())
    return PoseMotion(float(displacement[common_torso].max()) if torso_count >= 3 else float("inf"),
                      float(displacement[common_limb].max()) if limb_count else 0.0,
                      torso_count, limb_count)
