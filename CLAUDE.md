# Build & Boulder: working notes

Context for Claude sessions on this fork (phone or laptop). The event is on
Thu 24 Sep 2026, 18:30–21:30, at City Bouldering Aldgate, run by Tano. About 30
engineers in teams do bouldering challenges, relays and team games for points.
VLM Run (founder Jeremy Park) gave $25 of credits to build with
`rock_climbing/`. We build from the phone, so keep answers short and give
commands that can be copied in one go.

## What the demo does
`rock_climbing/main.py` takes a still-camera video of a boulder route and works
out which holds were used, in what order, and how long the climb took.
1. **SAM 3.1** (`facebook/sam3.1`) segments the holds of `HOLD_COLOR` on 12
   sampled frames and votes them into one hold map.
2. **ViTPose+ Large** (`usyd-community/vitpose-plus-large`) tracks the
   climber's body keypoints on every frame.
3. A hand or foot resting on a hold for 0.5s or more counts as a contact.
   Contacts give the sequence, the moves, the limb usage and the time to top
   (both wrists on the top hold).
Both models run on the VLM Run gateway (`https://gateway.vlm.run/v1/openai`),
and both are paid per call.

## Setup
- Key: `VLMRUN_API_KEY`, either as an environment variable (preferred in the
  cloud) or in a `.env` at the repo root. Never print it, commit it or paste
  it into chat.
- Cloud or phone: `bash rock_climbing/setup_cloud.sh` (ffmpeg + pip packages).
- Laptop (Windows): `rock_climbing\.venv` already exists; run
  `.venv\Scripts\python main.py` from `rock_climbing\`.

## Run
1. Put clips in `rock_climbing/data/input/current/` (.mp4/.mov). In batch mode
   (the default) every clip counts as an attempt at the SAME route.
2. Edit `rock_climbing/config.py`: `HOLD_COLOR` (e.g. "yellow") and
   `ROUTE_GRADE`. For a cheap quick test, set `TRIM_SECONDS = 8.0`.
3. `cd rock_climbing && python main.py`
4. `python leaderboard.py` ranks the attempts (topped > fastest > fewest holds),
   adds a coach tip for each climber and writes `leaderboard.md`.

Output goes to `data/output/<timestamp>/`, with one subfolder per attempt in
batch mode: `*_climb.mp4` (annotated side-by-side video), `*_route.mp4`,
`holds.png`, `summary.txt`, `sequence.json`, `climb.json`, `hold_times.csv`,
`limb_usage.csv`, `metrics.json` (includes the cost). A batch run also writes
`comparison.json` and `sequences.txt`.

## Live spotter (laptop + camera)
`python live.py` (`--source 1` for another camera, `--source clip.mp4` to replay
a clip). It sends 1 frame per second to ViTPose (~0.3s round trip, ~$0.06/min).
At start it runs one SAM scan for the holds and the floor, which takes ~1 min
in the background. States: READY → CLIMBING → TOPPED / LANDED / DOWN →
ALERT after 30s lying on the mat following a fall. ALERT only shows a banner,
beeps and saves a snapshot; a person presses A. Keys: Q quit, A acknowledge,
R reset, S re-scan. `python live.py --selftest` checks the logic offline.
Events and alert snapshots go to `data/output/live/<stamp>/`.
Reception screen (design B): add `--serve 8780 --wall "Wall 2"` and open
http://localhost:8780/ (from the phone: http://<laptop-ip>:8780/). Demo with no
camera and no API: `python live.py --demo --fall-seconds 8 --serve 8780`.

## Filming rules (the analysis breaks without these)
- The phone stays completely still (prop it on a water bottle), and it stays in
  the same spot for every attempt.
- The whole route is in frame, from the start holds to the top hold.
- Use an easy route (VB/V0) with a distinctive hold colour.
- Trim the clip to just before the start and just after the top.
- Only one climber in frame, and nobody walking through the shot.

## Troubleshooting
- **0 holds found** → wrong `HOLD_COLOR`, or the clip isn't a climbing wall.
  Try another colour word ("pink", "purple", "black"). Holds are cached per
  clip, so set `REUSE_HOLDS = False` after changing settings.
- **Floor line detected wrong** → set `DETECT_FLOOR = False`.
- **Slow or expensive** → `TRIM_SECONDS`, or `EVERY_FRAME = False` with
  `VIDEO_FPS = 10.0`.
- **Render one attempt only** → `RENDER_ONLY = 2` (it still analyses them all).
- An HDR iPhone clip needs tone-mapping (`TONEMAP = "auto"`). If it fails,
  record in SDR ("Most Compatible" format) or convert the clip.

## Ideas to stand out (in order)
1. **Team leaderboard + coach tips** (`leaderboard.py`, already built): film
   every teammate on the same route, run the batch, then show the table.
2. Add an LLM "beta coach": feed a climber's `sequence.json` to an LLM for a
   3-sentence critique (which limb to move differently, where to skip holds).
3. "Ghost climber": overlay the fastest attempt's route on a slower attempt's
   video to show where the time went.

## People
- Hosts: Sasha, Kieran Martin (kieran@tano.ai), Sagar Shah (sagar@tano.ai).
  Tano is hiring.
- Jeremy Park, founder of VLM Run. Follow VLM Run and Jeremy on LinkedIn/X
  if we use the repo.
