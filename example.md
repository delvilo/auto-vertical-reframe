========================================================================
            CINEMATIC AUTO-REFRAME CLI RECIPE CHEAT SHEET
========================================================================

0. FIXED INITIAL CAMERA (固定初始鏡頭 / 鎖定開場主角與固定焦段)
------------------------------------------------------------------------
python auto_reframe.py input.mp4 output_fixed.mp4 \\
    --preset talking_head \\
    --lock-first-subject \\
    --fixed-zoom 1.10 \\
    --dead-zone 0.08 \\
    --pan-time 0.55 \\
    --post-restore

1. STANDARD TALKING HEAD (YouTube Shorts / TikTok / Reels)
------------------------------------------------------------------------
python auto_reframe.py input.mp4 output_vertical.mp4 \\
    --preset talking_head \\
    --conf 0.35 \\
    --lock-first-subject \\
    --post-restore

2. TWO-PERSON PODCAST / CONVERSATION (Group Composition)
------------------------------------------------------------------------
python auto_reframe.py podcast.mp4 output_podcast.mp4 \\
    --preset talking_head \\
    --two-person-framing \\
    --two-person-threshold 0.75 \\
    --switch-score-threshold 1.25 \\
    --min-subject-hold-frames 18

3. SPORTS / ACTION (Fast Tracking, Low Damping, Wide FOV)
------------------------------------------------------------------------
python auto_reframe.py skate.mp4 output_sports.mp4 \\
    --preset sports \\
    --classes person bicycle motorcycle \\
    --motion-response 0.18 \\
    --motion-damping 0.75 \\
    --max-step-x 16.0

4. HIGH-FIDELITY SALIENCY + DEBUG TELEMETRY HUD
------------------------------------------------------------------------
python auto_reframe.py footage.mp4 output_vertical.mp4 \\
    --preset talking_head \\
    --saliency-model deepgazemsdb \\
    --saliency-device auto \\
    --post-restore \\
    --save-debug-preview \\
    --debug-path output_debug_hud.mp4
========================================================================
