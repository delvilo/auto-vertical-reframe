"""Build the automatic pose / inexpensive / temporal-neural cascade."""
from reframe.config import AppConfig
from reframe.saliency.cascade import CascadeSaliencyService


def build_saliency_helper(args: AppConfig) -> CascadeSaliencyService:
    from reframe.saliency.backends.deepgazemr import DeepGazeMRSaliencyHelper
    backend = DeepGazeMRSaliencyHelper(device=args.saliency_device,
                                      max_side=args.saliency_max_side,
                                      trust_repo=args.saliency_trust_repo)
    backend.max_failures = args.saliency_max_failures
    backend.use_amp = args.saliency_amp
    return CascadeSaliencyService(
        backend, args.saliency_interval, args.saliency_max_side, args.saliency_ema,
        lock_first_subject=args.lock_first_subject,
        two_person_framing=args.two_person_framing,
        keypoint_conf=args.keypoint_conf,
        pose_max_age=max(args.cue_interval, 1 / max(args.runtime_fps, 1)),
    )
