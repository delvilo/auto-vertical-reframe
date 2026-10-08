"""Select an existing backend without importing unused neural implementations."""
from reframe.config import AppConfig
from reframe.saliency.service import SaliencyService


def build_saliency_helper(args: AppConfig) -> SaliencyService:
    if args.saliency_model == "handcrafted":
        from reframe.saliency.backends.handcrafted import HandcraftedSaliencyHelper
        backend = HandcraftedSaliencyHelper()
    elif args.saliency_model in {"auto", "deepgazemsdb"}:
        from reframe.saliency.backends.deepgazemsdb import DeepGazeMSDBSaliencyHelper
        backend = DeepGazeMSDBSaliencyHelper(
            device=args.saliency_device, center_bias=args.saliency_center_bias,
            screen_inches=args.saliency_screen_inches,
            viewing_distance_cm=args.saliency_viewing_distance_cm,
            output_size=(args.output_width, args.output_height),
            pixel_per_dva=args.saliency_pixel_per_dva,
            allow_fallback=args.saliency_model == "auto",
            max_failures=args.saliency_max_failures, use_amp=args.saliency_amp,
        )
    else:
        raise ValueError(f"Unknown saliency backend: {args.saliency_model}")
    max_side = args.saliency_max_side if args.saliency_model == "handcrafted" else None
    return SaliencyService(backend, args.saliency_interval, max_side, args.saliency_ema)
