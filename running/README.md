# running

Measures a runner's cadence from a clip, times every foot strike, and draws the
returned keypoints back onto the video beside a live panel. Pose estimation runs
on
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
through the [VLM Run Gateway](https://www.vlm.run/gateway), so there are no
model weights to download.

For how the cadence is measured, see [cadence-explained.md](cadence-explained.md).

![A runner on a treadmill with a pose overlay on the left and a live cadence panel on the right](readme_images/running_demo_thumbnail.jpg)

## Run it

1. **Get an API key** at [app.vlm.run/sign-in](https://app.vlm.run/sign-in).

2. **Set it** in a `.env` at the repo root:

   ```bash
   cp ../.env.example ../.env
   # paste your key after VLMRUN_API_KEY=
   ```

3. **Create the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate running
   ```

4. **Add a clip** of someone running under `data/input/`, then point
   `INPUT_VIDEO` in [`config.py`](config.py) at it. Film from the side, with both
   ankles visible; a treadmill is ideal because the camera can stay still. For a
   handheld camera, set `FOOT_REFERENCE = "hip"`. Set `PANEL_KNEE_FEET` to the
   near leg.

5. **Run it** from this directory:

   ```bash
   python main.py
   ```

## Output

One timestamped directory per run under `data/output/`:

```
20260909-193259/
├── <clip>_pose.mp4     # H.264 overlay video + live panel
├── clearance.png       # both ankles' height, every strike marked, cadence under it
├── knee.png            # every detected knee, for debugging
├── steps.png           # step and contact time per step, the two feet separated
├── gait.json           # per-step timings + the full per-frame signals
├── summary.txt         # the gait report, human-readable
├── metrics.txt         # performance, with metrics.json beside it
├── poses.json          # the raw gateway response
└── run.json            # config + provenance
```

- Every knob lives in [`config.py`](config.py), and each run snapshots the ones
  it used into `run.json`.
- Converted MP4s and gateway responses are cached in `data/cache/`, keyed on the
  source file *and* the settings, so a changed setting never reuses a stale file.
- Both directories are gitignored, so nothing you record ends up in the repo.

## License

[Apache-2.0](../LICENSE).
