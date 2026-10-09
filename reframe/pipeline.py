from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
import cv2
from reframe.config import AppConfig, CLASS_IDS
from reframe.contracts import CameraState
from reframe.precrop import InferenceRegion
from reframe.geometry import compute_base_crop, current_crop_size, clamp, lerp
from reframe.perception.segmentation import SegmentationTracker
from reframe.perception.pose import YOLOPoseHelper, PoseCueCache, observe_poses
from reframe.saliency.factory import build_saliency_helper
from reframe.subjects import (SubjectRankingModel, load_speaker_segments, build_candidates,
                              choose_subject, choose_two_person_pair)
from reframe.camera import (reset_for_new_scene, pair_fits, build_pair_observation,
                            build_single_subject_observation, build_global_saliency_observation,
                            apply_camera_motion, crop_frame, regression_velocity)
from reframe.scenes import detect_scenes, InlineSceneDetector
from reframe.video_io import (DirectVideoWriter, LosslessWriter, build_video_filters,
                              run_ffmpeg_mux, iter_video_frames)
from reframe.debug import draw_debug
from reframe.timing import PhaseTimings
from reframe.segments import SEGMENT_SCHEMA_VERSION, SegmentStats, counter_snapshot


def process_video(args: AppConfig) -> None:
    started = time.perf_counter()
    pipeline_timings = PhaseTimings(("setup", "decode", "scene", "segmentation", "pose",
                                    "feedback", "saliency", "subjects", "camera",
                                    "render_write", "bookkeeping", "segment_logging", "finalization"),
                                   "setup", started)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    if args.save_debug_preview:
        debug_path = (
            Path(args.debug_path)
            if args.debug_path
            else output_path.with_name(output_path.stem + "_debug.mp4")
        )
        if debug_path.resolve() in {input_path.resolve(), output_path.resolve()}:
            raise ValueError("Debug output must differ from input and final output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    cap = cv2.VideoCapture(str(input_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if not math.isfinite(fps) or fps <= 0 or min(frame_w, frame_h) < 64:
        raise ValueError("Invalid video dimensions or FPS")
    region = InferenceRegion.from_frame(frame_w, frame_h, args.precrop)
    fallback_x, fallback_y = region.center
    logging.info("Inference region=%s bounds=%s source=%sx%s; rendering from original frames",
                 args.precrop or "full", region.bounds, frame_w, frame_h)
    args.runtime_fps = fps
    args.reference_dt = 30.0 / fps
    args.max_missed_frames = round(
        fps
        * (
            args.lost_hold_seconds
            if args.lost_hold_seconds is not None
            else args.max_missed_frames / 30.0
        )
    )
    args.min_subject_hold_frames = round(
        fps
        * (
            args.subject_hold_seconds
            if args.subject_hold_seconds is not None
            else args.min_subject_hold_frames / 30.0
        )
    )
    base_crop_w, base_crop_h = compute_base_crop(
        frame_w, frame_h, args.output_width, args.output_height
    )
    allowed_class_ids = [CLASS_IDS[name] for name in args.classes if name in CLASS_IDS]
    if not allowed_class_ids:
        raise ValueError("No valid classes selected for reframing.")

    speaker_segments = load_speaker_segments(args.speaker_json)
    if args.speaker_aware_mode:
        if any(
            not isinstance(seg, dict)
            or not {"start", "end", "track_id", "scene_index"} <= seg.keys()
            for seg in speaker_segments
        ):
            raise ValueError(
                "Speaker segments require start/end seconds, track_id and scene_index (1-based)"
            )
        if not speaker_segments:
            logging.warning(
                "Speaker-aware mode has no mapped segments and will not affect ranking"
            )
    if args.scene_method == "prepass":
        pipeline_timings.switch("scene")
        scene_start_set = set(
            detect_scenes(
                str(input_path),
                args.scene_threshold,
                args.min_scene_len,
                args.scene_downscale,
            )
        )
        pipeline_timings.switch("setup")
        inline_scene = None
    else:
        scene_start_set = set()
        inline_scene = InlineSceneDetector(args.min_scene_len, args.scene_threshold)

    with ExitStack() as model_resources:
        tracker = SegmentationTracker(args, allowed_class_ids)
        model_resources.callback(tracker.close)
        class_names = tracker.class_names
        saliency_helper = build_saliency_helper(args)
        model_resources.callback(saliency_helper.close)
        ranking_model = SubjectRankingModel()
        pose_helper = (PoseCueCache(YOLOPoseHelper(args), args.cue_interval, fps,
                                   tracking_max_age=args.seg_max_age)
                       if CLASS_IDS["person"] in allowed_class_ids else None)
        if pose_helper is not None:
            model_resources.callback(pose_helper.close)
        with (
            tempfile.TemporaryDirectory(dir=args.temp_dir) as tmpdir,
            ExitStack() as resources,
        ):
            tmpdir_path = Path(tmpdir)
            silent_video_path = tmpdir_path / "silent_vertical.mkv"
            debug_video_path = tmpdir_path / "debug_vertical.mkv"

            size = (args.output_width, args.output_height)
            writer = (
                DirectVideoWriter(
                    output_path,
                    input_path,
                    fps,
                    size,
                    args,
                    build_video_filters(args.post_restore),
                )
                if args.encode_mode == "direct"
                else LosslessWriter(str(silent_video_path), fps, size, args.ffmpeg_log_level)
            )
            resources.callback(writer.abort)
            if not writer.isOpened():
                raise RuntimeError("Could not create temporary video writer.")

            debug_writer = None
            final_debug_path = None
            if args.save_debug_preview:
                final_debug_path = (
                    Path(args.debug_path)
                    if args.debug_path
                    else output_path.with_name(output_path.stem + "_debug.mp4")
                )
                debug_writer = (
                    DirectVideoWriter(final_debug_path, input_path, fps, size, args)
                    if args.encode_mode == "direct"
                    else LosslessWriter(str(debug_video_path), fps, size, args.ffmpeg_log_level)
                )
                resources.callback(debug_writer.abort)
                if not debug_writer.isOpened():
                    raise RuntimeError("Could not create debug preview writer.")

            state = CameraState(
                crop_center_x=fallback_x,
                crop_center_y=fallback_y,
                zoom=args.min_zoom,
                target_center_x=fallback_x,
                target_center_y=fallback_y,
                target_zoom=args.min_zoom,
                is_scene_cut=True,
            )

            stats = {
                "frames_processed": 0,
                "scene_resets": 0,
                "subject_switches": 0,
                "frames_with_subject": 0,
                "frames_with_head_cues": 0,
                "frames_with_pose": 0,
                "frames_with_two_person": 0,
            }

            last_subject_key = None
            scene_index = 0

            frames = iter_video_frames(input_path)
            resources.callback(frames.close)
            actual_yolo_device = None

            def segment_snapshot(phase_overrides=None):
                seconds = pipeline_timings.snapshot()
                if phase_overrides:
                    seconds.update(phase_overrides)
                return counter_snapshot(stats, tracker.telemetry(), saliency_helper.telemetry(),
                                        pose_helper, seconds)

            segments = (SegmentStats(fps, args.stats_interval, segment_snapshot())
                        if args.stats_interval > 0 else None)

            def emit_segment(reason, resume_phase, phase_overrides=None):
                if segments is None or not segments.has_frames:
                    return
                # Keep report construction and terminal IO outside sampled frame
                # phases. The outer Summary still accounts for this overhead.
                pipeline_timings.switch("segment_logging")
                row = segments.finish(segment_snapshot(phase_overrides), reason)
                logging.info("Segment: %s", json.dumps(row, ensure_ascii=False, allow_nan=False))
                pipeline_timings.switch(resume_phase)

            final_phase_overrides = None

            while True:
                pipeline_timings.switch("decode")
                # Two scalar reads allow a cut-frame's decode/scene work to be
                # assigned to its new scene without copying model telemetry on
                # every frame. The EOF probe is likewise excluded from segments.
                before_frame_phases = ({key: pipeline_timings.seconds[key]
                                        for key in ("decode", "scene")}
                                       if segments is not None else None)
                try:
                    frame_idx, frame = next(frames)
                except StopIteration:
                    final_phase_overrides = before_frame_phases
                    break
                pipeline_timings.switch("scene")
                inference_frame = region.crop(frame)
                is_cut = (
                    inline_scene.update(frame, frame_idx)
                    if inline_scene
                    else frame_idx in scene_start_set
                )
                if is_cut:
                    emit_segment("scene", "scene", before_frame_phases)
                    tracker.reset()
                    if pose_helper is not None:
                        pose_helper.clear()
                    scene_index += 1
                    state.current_scene_index = scene_index
                    reset_for_new_scene(state, frame_w, frame_h, args.min_zoom)
                    state.crop_center_x = state.target_center_x = fallback_x
                    state.crop_center_y = state.target_center_y = fallback_y
                    saliency_helper.reset()
                    stats["scene_resets"] += 1
                    last_subject_key = None

                context = region.context(frame_idx, (frame_idx - 1) / fps, scene_index)
                pipeline_timings.switch("segmentation")
                tracks = tracker.track(inference_frame, context)
                actual_yolo_device = tracker.actual_device
                pipeline_timings.switch("pose")
                local_observations = observe_poses(inference_frame, tracks, pose_helper, state,
                                                   context, args.cue_top_k,
                                                   tracking_max_age=args.seg_max_age)
                pipeline_timings.switch("feedback")
                primary_id = state.lock_track_id if state.lock_track_id is not None else state.tracked_id
                tracker.feedback(local_observations, preferred_track_id=primary_id)
                pipeline_timings.switch("saliency")
                saliency = saliency_helper.process(inference_frame, local_observations,
                                                   preferred_track_id=primary_id)
                pipeline_timings.switch("subjects")
                observations = region.to_source(local_observations)
                saliency_map = saliency.map
                candidates = build_candidates(
                    observations=observations, saliency_map=saliency_map,
                    ranking_model=ranking_model, class_names=class_names, state=state, fps=fps,
                    speaker_segments=speaker_segments if args.speaker_aware_mode else [],
                    saliency_bounds=region.bounds,
                    tracking_max_age=args.seg_max_age,
                )

                subject = choose_subject(
                    candidates,
                    state,
                    args.lock_first_subject,
                    args.min_subject_hold_frames,
                    args.switch_score_threshold,
                    args.max_missed_frames,
                )

                pair = None
                if args.two_person_framing:
                    is_pair_active = state.two_person_active_frames > 0
                    pair = choose_two_person_pair(
                        candidates,
                        args.two_person_threshold,
                        is_pair_active,
                    )

                pipeline_timings.switch("camera")
                if pair is not None and not pair_fits(
                    pair, base_crop_w, base_crop_h, args.fixed_zoom or args.min_zoom
                ):
                    pair = None

                if pair is not None:
                    stats["frames_with_two_person"] += 1
                    state.two_person_active_frames += 1
                    observation = build_pair_observation(
                        pair[0],
                        pair[1],
                        base_crop_w,
                        base_crop_h,
                        frame_w,
                        frame_h,
                        args.min_zoom,
                        args.max_zoom,
                    )
                    crop_w, crop_h = apply_camera_motion(
                        state,
                        observation,
                        args,
                        base_crop_w,
                        base_crop_h,
                        frame_w,
                        frame_h,
                    )
                    state.missed_frames = 0
                    state.tracked_id = None
                    state.tracked_cls_id = None
                    state.frames_since_subject_switch = 0
                    state.last_framing_cx = None
                    state.last_framing_cy = None
                    state.last_subject_key = None
                    state.motion_history.clear()
                    state.framing_vx = 0.0
                    state.framing_vy = 0.0

                elif subject is not None:
                    state.two_person_active_frames = 0
                    previous_subject_key = (state.tracked_id, state.tracked_cls_id)
                    current_subject_key = (subject.track_id, subject.cls_id)
                    if (
                        last_subject_key is not None
                        and current_subject_key != last_subject_key
                    ):
                        stats["subject_switches"] += 1
                    last_subject_key = current_subject_key

                    stats["frames_with_subject"] += 1
                    if subject.head_box is not None:
                        stats["frames_with_head_cues"] += 1
                    if subject.has_pose:
                        stats["frames_with_pose"] += 1

                    if (
                        args.lock_first_subject
                        and state.lock_track_id is None
                        and subject.track_id is not None
                    ):
                        state.lock_track_id = subject.track_id
                        state.lock_cls_id = subject.cls_id

                    observation = build_single_subject_observation(
                        subject=subject,
                        base_crop_w=base_crop_w,
                        base_crop_h=base_crop_h,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        min_zoom=args.min_zoom,
                        max_zoom=args.max_zoom,
                    )

                    subject_key = (subject.track_id, subject.cls_id)
                    if state.last_subject_key != subject_key or state.missed_frames:
                        state.motion_history.clear()
                    now = (frame_idx - 1) / fps
                    state.motion_history.append(
                        (now, subject.framing_cx, subject.framing_cy)
                    )
                    while (
                        state.motion_history
                        and now - state.motion_history[0][0] > args.prediction_window
                    ):
                        state.motion_history.popleft()
                    vx, vy = regression_velocity(state.motion_history)
                    state.framing_vx = clamp(vx, -frame_w, frame_w)
                    state.framing_vy = clamp(vy, -frame_h, frame_h)
                    lookahead = args.lookahead_seconds * clamp(
                        observation.confidence, 0.0, 1.0
                    )
                    observation.center_x = clamp(
                        observation.center_x + state.framing_vx * lookahead,
                        0.0,
                        float(frame_w),
                    )
                    observation.center_y = clamp(
                        observation.center_y + state.framing_vy * lookahead * 0.5,
                        0.0,
                        float(frame_h),
                    )

                    state.last_framing_cx = subject.framing_cx
                    state.last_framing_cy = subject.framing_cy
                    state.last_subject_key = subject_key

                    crop_w, crop_h = apply_camera_motion(
                        state,
                        observation,
                        args,
                        base_crop_w,
                        base_crop_h,
                        frame_w,
                        frame_h,
                    )

                    state.tracked_id = subject.track_id
                    state.tracked_cls_id = subject.cls_id
                    state.missed_frames = 0
                    if current_subject_key == previous_subject_key:
                        state.frames_since_subject_switch += 1
                    else:
                        state.frames_since_subject_switch = 0
                else:
                    state.two_person_active_frames = 0
                    state.missed_frames += 1
                    state.motion_history.clear()
                    state.framing_vx = state.framing_vy = 0.0
                    observation = build_global_saliency_observation(
                        saliency_map=saliency_map,
                        base_crop_w=base_crop_w,
                        base_crop_h=base_crop_h,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        min_zoom=args.min_zoom,
                        max_zoom=args.max_zoom,
                        saliency_bounds=region.bounds,
                    )

                    if (
                        not state.is_scene_cut
                        and state.missed_frames <= args.max_missed_frames
                    ):
                        state.target_center_x = state.crop_center_x
                        state.target_center_y = state.crop_center_y
                        state.target_zoom = state.zoom
                        state.velocity_x = state.velocity_y = state.zoom_velocity = 0.0
                        crop_w, crop_h = current_crop_size(
                            base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
                        )
                    elif observation is not None:
                        state.tracked_id = state.tracked_cls_id = None
                        crop_w, crop_h = apply_camera_motion(
                            state,
                            observation,
                            args,
                            base_crop_w,
                            base_crop_h,
                            frame_w,
                            frame_h,
                        )
                    else:
                        crop_w, crop_h = current_crop_size(
                            base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
                        )
                        if state.missed_frames > args.max_missed_frames:
                            state.crop_center_x = lerp(
                                state.crop_center_x,
                                fallback_x,
                                1 - 0.97**args.reference_dt,
                            )
                            state.crop_center_y = lerp(
                                state.crop_center_y,
                                fallback_y,
                                1 - 0.97**args.reference_dt,
                            )
                            state.zoom = lerp(
                                state.zoom,
                                args.fixed_zoom or args.min_zoom,
                                1 - 0.95**args.reference_dt,
                            )
                            state.target_zoom = lerp(
                                state.target_zoom,
                                args.fixed_zoom or args.min_zoom,
                                1 - 0.95**args.reference_dt,
                            )
                        state.tracked_id = None
                        state.tracked_cls_id = None
                        state.frames_since_subject_switch = 0

                    state.crop_center_x = clamp(
                        state.crop_center_x,
                        crop_w / 2.0,
                        frame_w - crop_w / 2.0,
                    )
                    state.crop_center_y = clamp(
                        state.crop_center_y,
                        crop_h / 2.0,
                        frame_h - crop_h / 2.0,
                    )

                crop_w, crop_h = current_crop_size(
                    base_crop_w, base_crop_h, state.zoom, frame_w, frame_h
                )
                pipeline_timings.switch("render_write")
                cropped, crop_rect = crop_frame(
                    frame, state.crop_center_x, state.crop_center_y, crop_w, crop_h
                )
                clean_out = cv2.resize(
                    cropped,
                    (args.output_width, args.output_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                writer.write(clean_out)

                if debug_writer is not None:
                    dbg = draw_debug(
                        frame=frame,
                        crop_rect=crop_rect,
                        candidates=candidates,
                        subject=subject,
                        pair=pair,
                        state=state,
                        frame_idx=frame_idx,
                        total_frames=total_frames,
                        output_width=args.output_width,
                        output_height=args.output_height,
                    )
                    debug_writer.write(dbg)

                pipeline_timings.switch("bookkeeping")
                stats["frames_processed"] += 1
                if segments is not None:
                    segments.record_frame(frame_idx, scene_index, tracker.last_decision)
                    if segments.window_complete():
                        emit_segment("interval", "bookkeeping")
                if args.max_frames is not None and stats["frames_processed"] >= args.max_frames:
                    logging.info("Reached diagnostic frame limit: %s", args.max_frames)
                    break

                if frame_idx % 50 == 0:
                    saliency_telemetry = saliency_helper.telemetry()
                    logging.info(
                        "Processed %s/%s | scene=%s | zoom=%.2fx | tracked=%s | saliency=%s",
                        frame_idx,
                        total_frames if total_frames > 0 else "?",
                        state.current_scene_index,
                        state.zoom,
                        state.tracked_id,
                        saliency_telemetry.get("active_backend"),
                    )

            emit_segment("end", "bookkeeping", final_phase_overrides)
            pipeline_timings.switch("finalization")
            writer.release()
            if debug_writer is not None:
                debug_writer.release()

            actual_encoder = getattr(writer, "encoder", None)
            if args.encode_mode == "lossless":
                vf = build_video_filters(post_restore=args.post_restore)
                actual_encoder = run_ffmpeg_mux(
                    silent_video_path=str(silent_video_path),
                    source_input_path=str(input_path),
                    final_output_path=str(output_path),
                    video_encoder=args.video_encoder,
                    audio_bitrate=args.audio_bitrate,
                    crf=args.crf,
                    preset=args.preset_ffmpeg,
                    vf=vf,
                    nvenc_preset=args.nvenc_preset,
                    video_bitrate=args.video_bitrate,
                    ffmpeg_log_level=args.ffmpeg_log_level,
                )

                if debug_writer is not None and final_debug_path is not None:
                    run_ffmpeg_mux(
                        silent_video_path=str(debug_video_path),
                        source_input_path=str(input_path),
                        final_output_path=str(final_debug_path),
                        video_encoder=args.video_encoder,
                        audio_bitrate=args.audio_bitrate,
                        crf=args.crf,
                        preset=args.preset_ffmpeg,
                        vf=None,
                        nvenc_preset=args.nvenc_preset,
                        video_bitrate=args.video_bitrate,
                        ffmpeg_log_level=args.ffmpeg_log_level,
                    )

            saliency_telemetry = saliency_helper.telemetry()
            elapsed = pipeline_timings.switch(None) - started
            stage_seconds = {key: pipeline_timings.seconds[key]
                             for key in ("segmentation", "pose", "saliency", "render_write")}
            summary = {
                "preset": args.preset,
                "frames_processed": stats["frames_processed"],
                "stats_interval": args.stats_interval,
                "segment_schema_version": SEGMENT_SCHEMA_VERSION,
                "segments_emitted": segments.emitted if segments is not None else 0,
                "segment_pipeline_timing_seconds": (dict(segments.pipeline_timing_seconds)
                                                    if segments is not None else {}),
                "scene_resets": stats["scene_resets"],
                "frames_with_subject": stats["frames_with_subject"],
                "frames_with_head_cues": stats["frames_with_head_cues"],
                "frames_with_pose": stats["frames_with_pose"],
                "frames_with_two_person": stats["frames_with_two_person"],
                "subject_switches": stats["subject_switches"],
                "output_width": args.output_width,
                "output_height": args.output_height,
                "post_restore": args.post_restore,
                "precrop": args.precrop,
                "inference_bounds": region.bounds,
                "elapsed_seconds": elapsed,
                "processing_fps": stats["frames_processed"] / max(elapsed, 1e-9),
                "stage_wall_seconds": stage_seconds,
                # Since schema 2, feedback has its own phase instead of being
                # included in saliency. Nested saliency/backend timers must not
                # be added to these exclusive pipeline phases.
                "timing_schema_version": 2,
                "pipeline_timing_seconds": dict(pipeline_timings.seconds),
                "pipeline_timing_kind": "exclusive_host_wall",
                "pipeline_timing_scope": "through_writer_finalization_before_resource_cleanup",
                "encode_mode": args.encode_mode,
                "seg_model": args.seg_model,
                "seg_max_gap": args.seg_max_gap,
                "seg_max_age": args.seg_max_age,
                "yolo_device": actual_yolo_device,
                "pose_model": args.pose_model if pose_helper else None,
                "pose_device": pose_helper.helper.actual_device if pose_helper else None,
                "pose_rois_inferred": pose_helper.helper.rois_inferred if pose_helper else 0,
                "pose_rois_matched": pose_helper.helper.rois_matched if pose_helper else 0,
                "pose_cache_hits": pose_helper.cache_hits if pose_helper else 0,
                "pose_primary_only_frames": pose_helper.primary_only_frames if pose_helper else 0,
                "pose_full_scan_frames": pose_helper.full_scan_frames if pose_helper else 0,
                "pose_rois_skipped": pose_helper.rois_skipped if pose_helper else 0,
                "video_encoder_requested": args.video_encoder,
                "video_encoder_actual": actual_encoder,
                "scene_method": args.scene_method,
                "retina_masks": args.retina_masks,
                "saliency_active_backend": saliency_telemetry.get("active_backend"),
            }
            summary.update({f"saliency_{k}": v for k, v in saliency_telemetry.items()})
            summary.update({f"seg_{k}": v for k, v in tracker.telemetry().items()})
            logging.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
