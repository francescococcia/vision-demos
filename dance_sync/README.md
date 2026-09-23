# dance_sync

Scores how alike a group of dancers' poses are, one percentage per frame, and
draws the returned keypoints back onto the clip. Pose estimation runs on
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
through the [VLM Run Gateway](https://www.vlm.run/gateway), so there are no
model weights to download.

![Three dancers with pose overlays on the left and a sync-score panel on the right](readme_images/dance_demo_thumbnail.jpg)

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
   conda activate dance
   ```

4. **Add a clip** of two or more people dancing at `data/input/dance.MOV`, or
   point `INPUT_VIDEO` in [`config.py`](config.py) at your own file.

5. **Run it** from this directory:

   ```bash
   python main.py
   ```

The overlay video, the similarity plots and a `report.txt` land in a
timestamped directory under `data/output/`.

Every knob lives in [`config.py`](config.py). The pose estimation algorithm is
general and should work on other people's data, though some parameters were
tuned for nicer display on my clip.

For how the similarity number is built, see
[similarity-metric-explained.md](similarity-metric-explained.md).

## License

[Apache-2.0](../LICENSE).
