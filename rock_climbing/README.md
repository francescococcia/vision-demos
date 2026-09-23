# Rock Climbing + Computer Vision

Segments a boulder problem off the wall, follows the climber up it, and reads the
route back: which holds were used, in what order, and how long the send took.
Two models run on the [VLM Run Gateway](https://www.vlm.run/gateway):
[`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) for the holds and
the floor,
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
for the climber.

<p align="center">
  <img src="readme_images/rock_climbing_demo_thumbnail.jpg" width="600" alt="A completed boulder problem: the left panel holds a card comparing this attempt's sequence of holds against three others, the right panel the segmented route with the holds used lit in order">
</p>

## How to run this demo: tl;dr

1. Clone the [vision-demos](https://github.com/jeremyipark/vision-demos) repo.
2. Get an API key at [VLM Run](https://app.vlm.run/sign-in) and add it to your `.env`.
3. Record a video of yourself climbing, with your phone still and the whole route in frame.
4. Send the video to your computer and drop it in `data/input/current/`.
5. Update `HOLD_COLOR` in [`config.py`](config.py) to match your route.
6. Create the conda environment and run `python main.py`.

The details for each step are below.

## Resources

The rock climbing + computer vision demo
([LinkedIn video](https://www.linkedin.com/posts/jeremyipark_computervision-ai-ml-activity-7505763952740478976-cgKm)):

* Uses [ViTPose+ Large](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) for pose estimation
* Uses [SAM 3.1](https://vlm.run/gateway/models/facebook-sam3.1) to segment the bouldering holds
* Shows each hold light up as a hand or foot uses it
* Shows the sequence of holds used and the total time once the climb is done

Both models run on the VLM Run [Gateway](https://vlm.run/gateway), whose Model
Catalog gives you access to 22 vision models.

## Setup instructions

1. Clone the [vision-demos](https://github.com/jeremyipark/vision-demos) repo.
2. Sign up at [VLM Run](https://app.vlm.run/sign-in) to get an API key.
3. Copy your API key from the [main Overview dashboard](https://app.vlm.run/dashboard).
   The key is only shown once, when it's created, and then it's hidden, so copy
   it right away. You can also generate a new API key.
4. Add your API key to a `.env` file at the vision-demos root. From the
   `rock_climbing` folder, run:

   ```bash
   cp ../.env.example ../.env
   ```

   Then paste the key after `VLMRUN_API_KEY=`.

## Data collection

**tl;dr: keep your phone still using a tripod or a water bottle, and have the
entire route in frame.**

This rock climbing demo was based on a video taken with a tripod, so it assumes a
still camera. For best results:

* Have the entire route visible in frame, from the first hold to the last hold.
* Pick an easy route. For these demo videos, I typically go with a VB or V0.
* Use a tripod, or simply set your phone on a water bottle. Someone can also
  hold the camera, but they should try to stay as still as possible.
* Try to avoid recording other people, and watch out for anyone who might walk
  through the frame.

<p align="center">
  <img src="readme_images/rock_climbing_video_setup.jpg" width="320" alt="A climber at the start of a green route, filmed in portrait from a still phone; every green hold from the floor to the finish is in frame">
  <br>
  <em>Example framing of the rock climbing video. Note that all holds are in frame.</em>
</p>

In short, to create the minimal reproducible example of my demo:

1. Go to an easy VB/V0 route.
2. Put your phone on a water bottle.
3. Have the entire route in frame.
4. Record yourself completing the route.
5. Trim the video so it starts right before you begin the route and ends right
   after you finish it (this saves on inference time 🙂).
6. Send the video to your computer.

## Code instructions

**tl;dr: clone the vision-demos repo, point Claude or another coding agent at the
repo, and describe how you want to update the project.**

1. Add your trimmed video to the input folder:
   `vision-demos/rock_climbing/data/input/current/`
2. Create the conda environment (from the `rock_climbing` folder):

   ```bash
   conda env create -f environment.yml
   conda activate rock_climbing
   ```

3. Run the program:

   ```bash
   python main.py
   ```

**NOTE:** the hold color is currently a variable in [`config.py`](config.py).
Update `HOLD_COLOR` to match your route so that SAM 3.1 knows which holds to
segment (or ask your coding agent to do this).

```python
HOLD_COLOR = "green"                     # config.py
HOLD_PROMPT = "{color} climbing hold"
ROUTE_GRADE = "VB"                       # metadata; recorded, not drawn
```

Point it at another colour and the demo follows a different problem up the same
wall, with no retraining and no new model. That is what SAM 3.1 buys over a
detector fine-tuned on one gym's holds.

Every other knob lives in [`config.py`](config.py), and each run snapshots the
ones it used into `run.json`.

### Comparing several attempts

Every clip in `data/input/current/` is read as another attempt at the same route.
Drop one in and it behaves like a single run; drop four in and each render ends
on a card comparing its holds against the other three. Set `BATCH_MODE = False`
to run one clip, named by `INPUT_VIDEO`.

This is another reason to keep the camera still: holds are detected once per clip
and reused for every frame of it, and in batch mode the clips are compared to each
other hold for hold, so the camera should not move within a take or between takes.

For how the route is read and the attempts are aligned, see
[route-reading-explained.md](route-reading-explained.md).

## Output

In batch mode, one timestamped directory holding a subdirectory per clip:

```
20260915-153000/
├── comparison.json      # every attempt, aligned onto one numbering
├── sequences.txt        # the same thing, human-readable
├── IMG_8842/            # everything below, per clip
└── …
```

Otherwise one timestamped directory per run under `data/output/`:

```
20260914-231204/
├── climbing_climb.mp4   # the pair, side by side, with the original audio
├── climbing_route.mp4   # the right panel alone
├── holds.png            # the detected route on a clean frame; check this first
├── climb.json           # order, timings, per-hold contact, utilization
├── hold_times.csv       # one row per hold: order, limbs, timings
├── limb_usage.csv       # one row per (limb, hold): seconds and share
├── summary.txt          # the climb, human-readable
├── sequence.json        # the sequence, the moves, and the other attempts
├── holds.json           # every hold's mask polygon and bbox, normalized
├── poses.json           # the raw gateway response
├── metrics.txt          # throughput and cost, with metrics.json beside it
└── run.json             # config + provenance
```

Converted MP4s, segmentation results and pose responses are cached in
`data/cache/`, keyed on the source file *and* the settings that shaped them, so a
changed prompt never reuses a stale route. Both directories are gitignored.

If a run comes out wrong, look at `holds.png` first: nearly everything
downstream is a consequence of it. The knob for each symptom is in
[route-reading-explained.md](route-reading-explained.md).

## License

[Apache-2.0](../LICENSE).
