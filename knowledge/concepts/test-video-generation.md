---
title: "Test Video Generation"
aliases: [synthetic demo video, test.mp4, inference demo]
tags: [visualization, inference, moving-mnist, video-generation]
sources:
  - "daily/2026-04-18.md"
created: 2026-04-18
updated: 2026-04-18
word_count: 250
---

# Test Video Generation

**Test video generation** creates a synthetic video (`test.mp4`) for demonstrating real-time inference on the CluSTAR model. The video contains a single MNIST digit transitioning between action classes with smooth interpolations.

## Video Structure (Final Version)

**Duration**: 210 frames at 10fps = 21 seconds  
**Canvas**: 64×64 grayscale  
**Digit count**: 1  
**Actions**: moving → spinning → stationary (3 segments)

Each action segment:
- **60 steady frames** at constant velocity/rotation (matches `seq_length=60`)
- **15 transition frames** with smoothstep interpolation to next action's parameters
- Total per segment: 75 frames (except last, no transition out)

## Smoothstep Transitions

Velocity and rotation parameters interpolate via smoothstep between actions:

```
t_norm = (frame - transition_start) / transition_length
t_smooth = 3*t_norm² - 2*t_norm³  # smoothstep
param = param_old * (1 - t_smooth) + param_new * t_smooth
```

This avoids abrupt jumps that would create unrealistic motion.

## Metadata JSON

Generated alongside the video: `test_meta.json`

```json
{
  "fps": 10.0,
  "actions": ["moving", "spinning", "stationary"],
  "schedule": [
    {"action": "moving",     "start": 0,  "end": 59},
    {"action": "transition", "start": 60, "end": 74},
    {"action": "spinning",   "start": 75, "end": 134},
    ...
  ],
  "frame_labels": ["moving", "moving", ..., "spinning", ...]
}
```

Transition frames labeled as the action being transitioned **toward** (the action the model should be picking up).

## Implementation

Script: `clustar/scripts/generate_test_video.py`  
Dependencies: `imageio[ffmpeg]` (lightweight mp4 writer, no system ffmpeg required)  
Digit templates: reused from `MovingMNISTGenerator`
---ENDFILE---