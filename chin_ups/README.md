# chin_ups

Counts chin-up reps from a clip, times the pull on each one, and draws the
returned keypoints back onto the video beside a live stats panel. Pose
estimation runs on
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
through the [VLM Run Gateway](https://www.vlm.run/gateway), so there are no
model weights to download.

For how a rep is found and timed, see
[rep-counting-explained.md](rep-counting-explained.md).

![A chin-up at the top of the rep with a pose overlay on the left and a rep-timing panel on the right](readme_images/chin_ups_demo_thumbnail.jpg)

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
   conda activate chin_ups
   ```

4. **Add a clip** of someone doing chin-ups under `data/input/`, then point
   `INPUT_VIDEO` in [`config.py`](config.py) at it.

5. **Run it** from this directory:

   ```bash
   python main.py
   ```

## Output

One timestamped directory per run under `data/output/`:

```
20260909-150354/
├── <clip>_pose.mp4     # H.264 overlay video + stats panel
├── displacement.png    # the rep signal over time, every rep marked
├── reps.json           # per-rep timings + the full signal series
├── summary.txt         # the rep report, human-readable
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
