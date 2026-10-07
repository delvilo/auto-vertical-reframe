"""Select an existing backend without importing unused neural implementations."""
from reframe.config import AppConfig
from reframe.saliency.service import SaliencyService


def build_saliency_helper(args: AppConfig) -> SaliencyService:
    if args.saliency_model == "handcrafted":
        from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper
        backend = HandcraftedSaliencyHelper()
    elif args.saliency_model in {"auto", "deepgazemr"}:
        from reframe.saliency.backends.deepgazemr import DeepGazeMRSaliencyHelper
        backend = DeepGazeMRSaliencyHelper(device=args.saliency_device,
                                          max_side=args.saliency_max_side,
                                          trust_repo=args.saliency_trust_repo)
        backend.max_failures = args.saliency_max_failures
        backend.use_amp = args.saliency_amp
    else:
        raise ValueError(f"Unknown saliency backend: {args.saliency_model}")
    return SaliencyService(backend, args.saliency_interval, args.saliency_max_side, args.saliency_ema)
