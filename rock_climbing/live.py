r"""Live spotter: one frame per second to ViTPose, state and alerts on screen.

    python live.py                      # webcam 0 (use the phone as a webcam)
    python live.py --source 1           # another camera
    python live.py --source clip.mp4    # replay a recorded climb as if live
    python live.py --selftest           # check the fall logic, no camera, no API

Once at start, SAM 3.1 reads the holds (HOLD_COLOR) and the floor from a single
frame; the wall does not move, so that is not repeated. Then every second the
latest frame goes to ViTPose and the climber's state is updated locally:

    READY -> CLIMBING (both feet off the floor) -> TOPPED (both wrists on the top hold)
                     \-> LANDED  (back on the floor, upright)
                     \-> DOWN    (back on the floor, lying) -> ALERT after --fall-seconds

ALERT never calls anyone: it puts a red banner on screen, beeps and saves a
snapshot, and a person decides. Press A to acknowledge.

Keys: Q quit, A acknowledge alert, R reset climber, S re-scan the wall.
Cost: ~$0.001 per pose call, so ~$0.06 a minute at one frame per second.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config as cfg

# COCO-17 indices
NOSE, L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 9, 10, 11, 12, 15, 16
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
MIN_KPT_SCORE = 0.3
FALLBACK_FLOOR_Y = 0.92      # used when the floor could not be segmented
TOUCH_MARGIN = 0.02          # normalized distance around a hold's box that counts
COST_PER_POSE = 0.001


# ── pose parsing ─────────────────────────────────────────────────────────────

def parse_person(payload: dict, width: int, height: int):
    """The biggest person in the frame as (kpts Nx2 normalized, scores N, bbox) or None."""
    items = (payload.get("content") or {}).get("items") or []
    best, best_area = None, 0.0
    for item in items:
        kpts = np.asarray(item.get("kpts_xy") or [], dtype=np.float32).reshape(-1, 2)
        if len(kpts) < 17:
            continue
        bbox = np.asarray(item.get("bbox_xywh") or [0, 0, 0, 0], dtype=np.float32)
        if kpts.max() > 1.5:            # pixels, not normalized
            kpts = kpts / np.array([width, height], dtype=np.float32)
            bbox = bbox / np.array([width, height, width, height], dtype=np.float32)
        area = float(bbox[2] * bbox[3])
        if area > best_area:
            scores = np.asarray(item.get("kpts_score") or [1.0] * len(kpts), dtype=np.float32)
            best, best_area = (kpts, scores, bbox), area
    return best


# ── the climber's state ──────────────────────────────────────────────────────

class Spotter:
    """Turns one pose per second into READY / CLIMBING / TOPPED / LANDED / DOWN / ALERT."""

    def __init__(self, *, fall_seconds: float, floor=None, holds=None):
        self.fall_seconds = fall_seconds
        self.floor = floor
        self.holds = holds or []
        self.reset()

    def reset(self):
        self.state = "READY"
        self.climb_start = self.climb_time = self.down_since = None
        self.touched: list[int] = []
        self.events: list[dict] = []

    def floor_y(self, x: float) -> float:
        if self.floor is not None:
            y = self.floor.top_at(x)
            if y < 1.0:
                return y
        return FALLBACK_FLOOR_Y

    def feet_off(self, kpts, ok) -> bool:
        feet = [i for i in (L_ANK, R_ANK) if ok[i]]
        if len(feet) < 2:
            return False
        clearance = cfg.FLOOR_CLEARANCE + cfg.ANKLE_TO_TOE_OFFSET
        return all(kpts[i, 1] < self.floor_y(kpts[i, 0]) - clearance for i in feet)

    @staticmethod
    def lying(kpts, ok, bbox) -> bool:
        """Torso closer to horizontal than vertical, or a person wider than tall."""
        if ok[L_SH] and ok[R_SH] and ok[L_HIP] and ok[R_HIP]:
            sh, hip = (kpts[L_SH] + kpts[R_SH]) / 2, (kpts[L_HIP] + kpts[R_HIP]) / 2
            dx, dy = abs(sh[0] - hip[0]), abs(sh[1] - hip[1])
            if dx + dy > 0.02:
                return dx > dy
        return bbox[2] > 1.3 * bbox[3] if bbox[3] > 0 else False

    def holds_under(self, kpts, ok, limbs) -> set[int]:
        hit = set()
        for i in limbs:
            if not ok[i]:
                continue
            x, y = kpts[i]
            for h in self.holds:
                bx, by, bw, bh = h["bbox"]
                if bx - TOUCH_MARGIN <= x <= bx + bw + TOUCH_MARGIN and \
                        by - TOUCH_MARGIN <= y <= by + bh + TOUCH_MARGIN:
                    hit.add(h["id"])
        return hit

    def log(self, now: float, event: str, **extra):
        self.events.append({"t": round(now, 2), "event": event, **extra})

    def update(self, person, now: float) -> str:
        if self.state == "ALERT":
            return self.state                      # held until a person acknowledges
        if person is None:
            return self.state
        kpts, scores, bbox = person
        ok = scores >= MIN_KPT_SCORE
        off, flat = self.feet_off(kpts, ok), self.lying(kpts, ok, bbox)

        if self.state in ("READY", "LANDED") and off:
            self.state, self.climb_start, self.climb_time, self.touched = \
                "CLIMBING", now, None, []
            self.log(now, "start")
        if self.state == "CLIMBING":
            for hold_id in sorted(self.holds_under(kpts, ok, (L_WR, R_WR, L_ANK, R_ANK))):
                if hold_id not in self.touched:
                    self.touched.append(hold_id)
            top = max((h["id"] for h in self.holds), default=None)
            if top is not None and self.holds_under(kpts, ok, (L_WR,)) & \
                    self.holds_under(kpts, ok, (R_WR,)) & {top}:
                self.state, self.climb_time = "TOPPED", now - self.climb_start
                self.log(now, "topped", seconds=round(self.climb_time, 1), holds=self.touched)
            elif not off:
                self.climb_time = now - self.climb_start
                if flat:
                    self.state, self.down_since = "DOWN", now
                    self.log(now, "fall", seconds=round(self.climb_time, 1))
                else:
                    self.state = "LANDED"
                    self.log(now, "landed", seconds=round(self.climb_time, 1))
        elif self.state == "TOPPED" and not off:
            self.state = "DOWN" if flat else "LANDED"
            self.down_since = now if flat else None
            self.log(now, "fall" if flat else "landed")
        elif self.state == "DOWN":
            if not flat:
                self.state, self.down_since = "LANDED", None
                self.log(now, "got_up")
            elif now - self.down_since >= self.fall_seconds:
                self.state = "ALERT"
                self.log(now, "alert", down_seconds=round(now - self.down_since, 1))
        return self.state

    def acknowledge(self, now: float):
        if self.state == "ALERT":
            self.log(now, "acknowledged")
            self.state, self.down_since = "READY", None


# ── camera + gateway workers ─────────────────────────────────────────────────

class Camera(threading.Thread):
    """Keeps only the newest frame. A file is paced to its own fps, as if live."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera/video: {source!r}")
        self.is_file = isinstance(source, str)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame, self.ended, self.lock = None, False, threading.Lock()

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                self.ended = True
                return
            with self.lock:
                self.frame = frame
            if self.is_file:
                time.sleep(1.0 / self.fps)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def jpeg_b64(frame, height: int = 720) -> str:
    h, w = frame.shape[:2]
    if h > height:
        frame = cv2.resize(frame, (int(w * height / h), height))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii")


def request_pose(client, frame):
    h, w = frame.shape[:2]
    response = client.chat.completions.create(
        model=cfg.POSE_MODEL,
        messages=[{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(frame)}"}}]}],
        response_format={"type": "json_object"},
        extra_body={"method": "pose"},
    )
    return parse_person(json.loads(response.choices[0].message.content), w, h)


def scan_wall(client, frame):
    """One SAM pass for the holds and one for the floor, on a single frame."""
    from src import floor as floor_mod
    from src import holds as holds_mod
    per_frame, _ = holds_mod.segment_frames(
        client, [(0, frame)], model=cfg.HOLD_MODEL,
        prompt=cfg.HOLD_PROMPT.format(color=cfg.HOLD_COLOR),
        min_score=cfg.HOLD_MIN_SCORE, workers=1)
    holds = holds_mod.consensus(
        per_frame, iou_threshold=cfg.HOLD_IOU_THRESHOLD, min_appearance=0.0,
        raster_size=cfg.HOLD_RASTER_SIZE, vote_fraction=cfg.HOLD_VOTE_FRACTION,
        min_area_px=cfg.HOLD_MIN_AREA_PX, fill_holes=cfg.HOLD_FILL_HOLES)
    holds, _ = holds_mod.assign_ids(holds, band=cfg.HOLD_NUMBER_BAND)
    floor = None
    if cfg.DETECT_FLOOR:
        floor, _ = floor_mod.segment(
            client, [(0, frame)], model=cfg.HOLD_MODEL, prompt=cfg.FLOOR_PROMPT,
            resolution=cfg.FLOOR_EDGE_RESOLUTION, min_score=cfg.FLOOR_MIN_SCORE,
            min_area=cfg.FLOOR_MIN_AREA, workers=1)
    return holds, floor


# ── drawing ──────────────────────────────────────────────────────────────────

STATE_COLOR = {"READY": (200, 200, 200), "CLIMBING": (255, 200, 0), "TOPPED": (0, 220, 0),
               "LANDED": (0, 200, 255), "DOWN": (0, 140, 255), "ALERT": (0, 0, 255)}


def draw(frame, spotter: Spotter, person, info: dict, now: float):
    h, w = frame.shape[:2]
    px = lambda p: (int(p[0] * w), int(p[1] * h))
    for hold in spotter.holds:
        x, y, bw, bh = hold["bbox"]
        used = hold["id"] in spotter.touched
        cv2.rectangle(frame, px((x, y)), px((x + bw, y + bh)),
                      (0, 255, 0) if used else (200, 200, 200), 3 if used else 1)
        cv2.putText(frame, str(hold["id"]), px((x, y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    xs = np.linspace(0, 1, 64)
    ys = [spotter.floor_y(x) for x in xs]
    cv2.polylines(frame, [np.array([px((x, y)) for x, y in zip(xs, ys)], np.int32)],
                  False, (120, 230, 255), 2, cv2.LINE_AA)
    if person is not None:
        kpts, scores, _ = person
        ok = scores >= MIN_KPT_SCORE
        for a, b in BONES:
            if ok[a] and ok[b]:
                cv2.line(frame, px(kpts[a]), px(kpts[b]), (255, 0, 255), 3, cv2.LINE_AA)
        for i in range(len(kpts)):
            if ok[i]:
                cv2.circle(frame, px(kpts[i]), 4, (255, 255, 255), -1)

    state = spotter.state
    if state == "CLIMBING":
        detail = f"{now - spotter.climb_start:5.1f}s  holds {len(spotter.touched)}"
    elif state in ("TOPPED", "LANDED") and spotter.climb_time is not None:
        detail = f"{spotter.climb_time:.1f}s  holds {len(spotter.touched)}"
    elif state in ("DOWN", "ALERT") and spotter.down_since is not None:
        detail = f"down {now - spotter.down_since:4.0f}s / {spotter.fall_seconds:.0f}s"
    else:
        detail = ""
    cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(frame, f"{state}  {detail}", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                STATE_COLOR.get(state, (255, 255, 255)), 2, cv2.LINE_AA)
    meta = (f"{info['status']}  |  pose {info['latency']:.2f}s  |  "
            f"calls {info['calls']}  ~${info['calls'] * COST_PER_POSE:.3f}")
    cv2.putText(frame, meta, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    if state == "ALERT":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
        cv2.putText(frame, "CHECK ON THE CLIMBER - press A when handled",
                    (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return frame


def beep():
    try:
        import winsound
        winsound.Beep(1200, 400)
    except Exception:
        print("\a", end="", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between pose calls")
    ap.add_argument("--fall-seconds", type=float, default=30.0)
    ap.add_argument("--no-scan", action="store_true", help="skip the SAM wall scan")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from openai import OpenAI
    from src.env import load_api_key
    key, _ = load_api_key(cfg.PROJECT_DIR)
    client = OpenAI(api_key=key, base_url=cfg.GATEWAY_BASE_URL, timeout=30)

    source = int(args.source) if args.source.isdigit() else args.source
    cam = Camera(source)
    cam.start()
    while cam.latest() is None:
        if cam.ended:
            raise SystemExit("No frames from the source.")
        time.sleep(0.05)

    spotter = Spotter(fall_seconds=args.fall_seconds)
    info = {"status": "starting", "latency": 0.0, "calls": 0}
    shared = {"person": None, "scan": not args.no_scan, "stop": False}
    out_dir = cfg.OUTPUT_DIR / "live" / datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def worker():
        next_at = time.perf_counter()
        while not shared["stop"]:
            if shared["scan"]:
                shared["scan"] = False
                info["status"] = f"scanning wall ({cfg.HOLD_COLOR})..."
                try:
                    spotter.holds, spotter.floor = scan_wall(client, cam.latest())
                    info["status"] = f"{len(spotter.holds)} {cfg.HOLD_COLOR} holds" + \
                        (", floor found" if spotter.floor else ", floor fallback")
                except Exception as exc:
                    info["status"] = f"scan failed: {exc}"[:80]
            if time.perf_counter() < next_at:
                time.sleep(0.02)
                continue
            next_at = time.perf_counter() + args.interval
            frame = cam.latest()
            try:
                t = time.perf_counter()
                person = request_pose(client, frame)
                info["latency"], info["calls"] = time.perf_counter() - t, info["calls"] + 1
            except Exception as exc:
                info["status"] = f"pose error: {exc}"[:80]
                continue
            shared["person"] = person
            before = spotter.state
            now = time.perf_counter() - t0
            spotter.update(person, now)
            if spotter.state != before:
                print(f"[{now:7.1f}s] {before} -> {spotter.state}")
                if spotter.state == "ALERT":
                    cv2.imwrite(str(out_dir / f"alert_{int(now)}s.jpg"), frame)
                    beep()

    threading.Thread(target=worker, daemon=True).start()
    window = "Build & Boulder - live spotter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_beep = 0.0
    try:
        while True:
            frame = cam.latest()
            now = time.perf_counter() - t0
            if frame is not None:
                cv2.imshow(window, draw(frame, spotter, shared["person"], info, now))
            if spotter.state == "ALERT" and now - last_beep > 3:
                last_beep = now
                threading.Thread(target=beep, daemon=True).start()
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or (cam.ended and cam.is_file and key != 255):
                break
            if key == ord("a"):
                spotter.acknowledge(now)
            elif key == ord("r"):
                spotter.reset()
            elif key == ord("s"):
                shared["scan"] = True
    finally:
        shared["stop"] = True
        cv2.destroyAllWindows()
        (out_dir / "events.json").write_text(json.dumps(spotter.events, indent=2))
        print(f"events -> {out_dir / 'events.json'}  ({info['calls']} pose calls, "
              f"~${info['calls'] * COST_PER_POSE:.3f})")
    return 0


# ── self-test: the state machine on synthetic poses ──────────────────────────

def _pose(feet_y: float, lying: bool = False, wrists=None):
    k = np.zeros((17, 2), np.float32)
    if lying:
        k[:, 1] = feet_y
        k[L_SH], k[R_SH], k[L_HIP], k[R_HIP] = (0.3, feet_y), (0.3, feet_y - 0.01), \
            (0.5, feet_y), (0.5, feet_y - 0.01)
        k[L_ANK] = k[R_ANK] = (0.7, feet_y)
        bbox = np.array([0.3, feet_y - 0.05, 0.4, 0.08], np.float32)
    else:
        k[L_SH], k[R_SH] = (0.45, feet_y - 0.45), (0.55, feet_y - 0.45)
        k[L_HIP], k[R_HIP] = (0.47, feet_y - 0.2), (0.53, feet_y - 0.2)
        k[L_ANK], k[R_ANK] = (0.46, feet_y), (0.54, feet_y)
        k[L_WR], k[R_WR] = wrists or ((0.4, feet_y - 0.6), (0.6, feet_y - 0.6))
        bbox = np.array([0.4, feet_y - 0.65, 0.2, 0.65], np.float32)
    return k, np.ones(17, np.float32), bbox


def selftest() -> int:
    holds = [{"id": 1, "bbox": [0.38, 0.55, 0.04, 0.04]},
             {"id": 2, "bbox": [0.58, 0.12, 0.04, 0.04]}]
    s = Spotter(fall_seconds=30, holds=holds)
    floor = 0.9
    steps = [(0, _pose(floor), "READY"),
             (1, _pose(0.7, wrists=((0.40, 0.57), (0.6, 0.1))), "CLIMBING"),
             (2, _pose(0.5), "CLIMBING"),
             (3, _pose(floor, lying=True), "DOWN")]
    steps += [(3 + i, _pose(floor, lying=True), "DOWN") for i in range(1, 30)]
    steps += [(33, _pose(floor, lying=True), "ALERT"),
              (34, _pose(floor), "ALERT")]          # held until acknowledged
    failures = 0
    for t, person, want in steps:
        got = s.update(person, float(t))
        if got != want:
            failures += 1
            print(f"FAIL t={t}: want {want}, got {got}")
    s.acknowledge(35.0)
    checks = [(s.state == "READY", "acknowledge returns to READY"),
              (1 in s.touched, "hold 1 touched while climbing")]
    # A clean climb: up, top with both hands, land on feet.
    s2 = Spotter(fall_seconds=30, holds=holds)
    s2.update(_pose(floor), 0)
    s2.update(_pose(0.7), 1)
    top = ((0.59, 0.13), (0.61, 0.13))
    checks += [(s2.update(_pose(0.5, wrists=top), 5) == "TOPPED", "both wrists on top hold -> TOPPED"),
               (s2.update(_pose(floor), 7) == "LANDED", "landing upright -> LANDED"),
               (abs(s2.climb_time - 4.0) < 1e-6, "climb time 4.0s kept after landing")]
    # Sitting on the mat after a normal landing must never alert.
    s3 = Spotter(fall_seconds=30)
    s3.update(_pose(floor), 0)
    for t in range(1, 60):
        s3.update(_pose(floor, lying=True), float(t))
    checks.append((s3.state == "READY", "lying on the mat without a fall never alerts"))
    for ok, name in checks:
        failures += not ok
        print(("ok   " if ok else "FAIL ") + name)
    print("selftest passed" if not failures else f"selftest: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
