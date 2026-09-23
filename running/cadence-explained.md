# How the cadence is measured

1. Run pose estimation on every frame to get the body's keypoints.
2. Measure each ankle's height above that ankle's own stance level, in units of
   the runner's leg length.
3. Count a foot strike where the height crosses down through a threshold.
4. Time cadence on strides (same foot to the same foot), not on alternating steps.

Code: [`src/gait.py`](src/gait.py). Every parameter, and why each default is what
it is: [`config.py`](config.py). Per-step numbers land in `gait.json` and
`summary.txt`.

Everything below uses the 17 COCO keypoints that
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
returns per frame. No extra model, no markers.

One thing to know before any of it: **x is normalized by the frame width and y by
its height**, so on anything but a square frame they are not in the same unit.
`gait.analyze` scales x by the frame's aspect ratio before any angle or length is
measured, since a thigh and a shank are rigid segments whose length can't depend
on which way the limb points (mixed units, a tilted shank measured 45% longer
than an upright one; scaled, within 2%). Cadence doesn't depend on this — ankle
*height* is a single axis — but every knee number does.

## The signal

```
height[foot] = (ground[foot] - ankle_y[foot]) / leg_length
```

- COCO-17 ends at the ankles: no heel, no toes. A "foot strike" is detected from
  the ankle, several centimetres above the thing that hits the ground.
- The zero cannot be the floor, so `ground` is a *calibrated* stance level: the
  90th percentile of that ankle's height, one value per leg (the two ankles
  don't project to the same height on a side-on shot).
- `leg_length` is median thigh + median shank, each measured as a segment, not
  an average of `|hip_y - ankle_y|` — the knee is bent for most of a running
  clip, so that average really measures how bent it happened to be, and used to
  run 8-11% short.
- The result is two interleaved waves, near zero while a foot is planted and
  peaking at mid-swing.

## Finding the steps

- A strike is the downward crossing of `CONTACT_FRACTION` of typical swing
  height, where "typical" is the median of the peaks so one bad frame can't move
  every threshold.
- A swing that never clears `SWING_PEAK_FRACTION` is ankle wobble in stance; a
  stance shorter than `MIN_CONTACT_SECONDS` is the signal clipping the threshold
  mid-swing. Both are absorbed, or one planted foot reports two or three strikes.
- Intervals are timed on **interpolated mid-stance**, not the threshold crossing
  the overlay flashes on.
- Cadence is `120 / mean(stride_time)`. A stride compares one foot to itself, so
  a per-leg landmark offset cancels — alternating step times don't have that
  property.

## The panel

- **Cadence**, live, as an exponential average over overlapping strides.
- **Avg cadence over time**, the mean so far, one vertex per step.
- **Avg knee shape at foot strike**, the near leg by default (`PANEL_KNEE_FEET`
  picks which); both legs are always written to `gait.json`, `summary.txt` and
  `knee.png`.
- **Real-time ankle visualization**: the two ankle points with the body removed,
  as a coverage cloud, a comet trail, and a current-frame dot, on equal-scale
  axes. Nothing is drawn ahead of the current frame.
- No per-foot cadence or symmetry figure on the panel — the feet alternate, so
  neither can have a higher rate than the other, and left/right comparisons from
  one side-on camera mix the runner with near/far geometry (they stay in the
  files with that caveat attached).

`knee.png` is the check on the knee: a bad pose is a limb pointing somewhere the
others do not, or a stray flexion line. `clearance.png` is the check on the
strikes.

## Limits

- **No foot.** Strike pattern (heel / midfoot / forefoot) and ankle angle are
  not available.
- **2D.** Overstriding, hip drop, knee valgus and pronation need another view.
- **No distance or speed** on a treadmill: the runner does not translate.
- **Knee flexion is a projected angle**, not anatomical. It agrees with anatomy
  where anatomy has a number (17.5°/22.5° at contact vs. a textbook 20-25°; a
  stance peak of 36°/44° vs. 40-45°), but the swing peak (82°/92° vs. a
  published 100-125° for faster running) can only be an under-read: the shank
  turns furthest out of the image plane exactly there. Read contact and stance
  as calibrated, the swing peak as a floor, and each leg against itself rather
  than the other.
- One subject. The signal reads `persons[0]`.
- Left/right labels follow the model. ViTPose can swap the legs for a cycle when
  they scissor in the image; `knee.png` is where that shows up.
