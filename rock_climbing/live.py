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
        self.events.append({"t": round(now, 2), "at": datetime.now().strftime("%H:%M:%S"),
                            "event": event, **extra})

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


# ── reception screen: a tiny web server over the spotter ─────────────────────

WEB_DIR = Path(__file__).resolve().parent / "web"


def serve(port: int, ctx: dict):
    """Serve web/index.html plus the JSON/JPEG API the reception page polls."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                page = WEB_DIR / "index.html"
                if not page.exists():
                    return self.send(404, b"web/index.html missing", "text/plain")
                return self.send(200, page.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self.send(200, json.dumps(ctx["snapshot"]()).encode(),
                                 "application/json")
            if path == "/api/frame.jpg":
                jpg = ctx["jpeg"]()
                return self.send(200 if jpg else 404, jpg or b"", "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/ack":
                ctx["ack"]()
            elif path == "/api/reset":
                ctx["reset"]()
            else:
                return self.send(404, b"not found", "text/plain")
            self.send(200, b'{"ok":true}', "application/json")

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def snapshot_of(spotter: Spotter, shared: dict, info: dict, clock: float, wall: str) -> dict:
    state = spotter.state
    climb = None
    if state == "CLIMBING" and spotter.climb_start is not None:
        climb = clock - spotter.climb_start
    elif spotter.climb_time is not None:
        climb = spotter.climb_time
    down = clock - spotter.down_since if (
        spotter.down_since is not None and state in ("DOWN", "ALERT")) else None
    person = shared.get("person")
    frame = shared.get("frame")
    return {
        "state": state, "clock": round(clock, 2),
        "climb_seconds": None if climb is None else round(climb, 1),
        "down_seconds": None if down is None else round(down, 1),
        "fall_seconds": spotter.fall_seconds, "hold_color": cfg.HOLD_COLOR,
        "holds": [{"id": h["id"], "bbox": [round(float(v), 4) for v in h["bbox"]]}
                  for h in spotter.holds],
        "touched": list(spotter.touched),
        "person": None if person is None else {
            "kpts": [[round(float(x), 4), round(float(y), 4)] for x, y in person[0]],
            "ok": [bool(v) for v in (person[1] >= MIN_KPT_SCORE)]},
        "frame": {"w": frame.shape[1] if frame is not None else 0,
                  "h": frame.shape[0] if frame is not None else 0,
                  "seq": shared.get("seq", 0)},
        "latency": round(info["latency"], 3), "calls": info["calls"],
        "cost": round(info["calls"] * COST_PER_POSE, 4), "status": info["status"],
        "events": spotter.events[-50:], "wall": wall,
    }


# ── demo feed: the real state machine on a scripted climb, no camera, no API ─

DEMO_HOLDS = [{"id": i + 1, "bbox": [x, y, 0.035, 0.045]} for i, (x, y) in enumerate(
    [(0.42, 0.72), (0.55, 0.62), (0.45, 0.52), (0.58, 0.42), (0.47, 0.32),
     (0.56, 0.22), (0.50, 0.12)])]


def demo_frame(w=1280, h=720):
    img = np.full((h, w, 3), (52, 58, 66), np.uint8)
    for i in range(0, w, 64):
        cv2.line(img, (i, 0), (i, int(h * 0.9)), (60, 66, 74), 1)
    cv2.rectangle(img, (0, int(h * 0.9)), (w, h), (95, 95, 95), -1)       # the mat
    rng = np.random.default_rng(7)
    for _ in range(40):                                                   # other routes
        cx, cy = int(rng.uniform(0.05, 0.95) * w), int(rng.uniform(0.05, 0.85) * h)
        color = [(60, 60, 220), (220, 120, 40), (200, 60, 200)][int(rng.integers(3))]
        cv2.circle(img, (cx, cy), 9, color, -1)
    for hold in DEMO_HOLDS:
        x, y, bw, bh = hold["bbox"]
        cv2.ellipse(img, (int((x + bw / 2) * w), int((y + bh / 2) * h)),
                    (int(bw * w / 2), int(bh * h / 2)), 0, 0, 360, (60, 200, 60), -1)
    return img


def demo_script(t: float, fall_seconds: float):
    """(pose, phase) at t seconds into the scripted loop: stand, climb, fall, lie."""
    lie_for = fall_seconds + 8
    if t < 4:
        return _pose(0.9), "standing"
    if t < 12:                                   # climb from floor to hold 5
        f = (t - 4) / 8
        feet = 0.72 - 0.35 * f
        hand = DEMO_HOLDS[min(int(f * 5) + 1, 5)]["bbox"]
        return _pose(feet, wrists=((hand[0] + 0.01, hand[1] + 0.02),
                                   (hand[0] + 0.02, hand[1] + 0.02))), "climbing"
    if t < 12 + lie_for:
        return _pose(0.9, lying=True), "lying on the mat"
    return _pose(0.9), "got up"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between pose calls")
    ap.add_argument("--fall-seconds", type=float, default=30.0)
    ap.add_argument("--no-scan", action="store_true", help="skip the SAM wall scan")
    ap.add_argument("--serve", type=int, default=0, metavar="PORT",
                    help="serve the reception screen (web/index.html) on this port")
    ap.add_argument("--wall", default="Wall 1", help="wall name shown at reception")
    ap.add_argument("--demo", action="store_true",
                    help="scripted climb + fall through the real logic; no camera, no API")
    ap.add_argument("--headless", action="store_true",
                    help="no window; write the annotated view to live.mp4")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    out_dir = cfg.OUTPUT_DIR / "live" / datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"status": "starting", "latency": 0.0, "calls": 0}
    shared = {"person": None, "frame": None, "seq": 0, "jpeg": b"",
              "scan": not (args.no_scan or args.demo), "stop": False}
    frame_lock = threading.Lock()
    t0 = time.perf_counter()
    clock = lambda: time.perf_counter() - t0

    if args.demo:
        spotter = Spotter(fall_seconds=args.fall_seconds, holds=[dict(h) for h in DEMO_HOLDS])
        cam = None
        info["status"] = "DEMO: scripted climb, no camera, no API"
    else:
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

    def latest_frame():
        return demo_frame() if cam is None else cam.latest()

    def publish(frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            shared["frame"], shared["jpeg"] = frame, buf.tobytes() if ok else b""
            shared["seq"] += 1

    def on_person(person, frame):
        shared["person"] = person
        before = spotter.state
        now = clock()
        spotter.update(person, now)
        if spotter.state != before:
            print(f"[{now:7.1f}s] {before} -> {spotter.state}")
            if spotter.state == "ALERT":
                cv2.imwrite(str(out_dir / f"alert_{int(now)}s.jpg"), frame)
                if not args.serve:
                    beep()

    def worker():
        next_at = time.perf_counter()
        demo_t0 = time.perf_counter()
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
            frame = latest_frame()
            publish(frame)
            if args.demo:
                t = time.perf_counter() - demo_t0
                person, phase = demo_script(t, args.fall_seconds)
                if phase == "got up" and spotter.state in ("READY", "LANDED"):
                    demo_t0 = time.perf_counter()            # loop the script
                info["status"] = f"DEMO: {phase}"
                info["latency"], info["calls"] = 0.0, info["calls"]
                on_person(person, frame)
                continue
            try:
                t = time.perf_counter()
                person = request_pose(client, frame)
                info["latency"], info["calls"] = time.perf_counter() - t, info["calls"] + 1
            except Exception as exc:
                info["status"] = f"pose error: {exc}"[:80]
                continue
            on_person(person, frame)

    def frames_live():
        """Between pose calls, keep the reception picture moving at ~5 fps."""
        while not shared["stop"] and cam is not None:
            frame = cam.latest()
            if frame is not None:
                publish(frame)
            time.sleep(0.2)

    ctx = {
        "snapshot": lambda: snapshot_of(spotter, shared, info, clock(), args.wall),
        "jpeg": lambda: shared["jpeg"],
        "ack": lambda: spotter.acknowledge(clock()),
        "reset": lambda: spotter.reset(),
    }
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=frames_live, daemon=True).start()
    server = None
    if args.serve:
        server = serve(args.serve, ctx)
        print(f"reception screen -> http://localhost:{args.serve}/")

    window = "Build & Boulder - live spotter"
    writer = None
    show_window = not (args.headless or args.serve)
    if args.headless:
        h, w = latest_frame().shape[:2]
        writer = cv2.VideoWriter(str(out_dir / "live.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 1000 / 30, (w, h))
    elif show_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_beep = 0.0
    try:
        while True:
            now = clock()
            key = 255
            if writer is not None or show_window:
                frame = latest_frame()
                if frame is not None:
                    shown = draw(frame, spotter, shared["person"], info, now)
                    if writer is not None:
                        writer.write(shown)
                    else:
                        cv2.imshow(window, shown)
            if show_window:
                if spotter.state == "ALERT" and now - last_beep > 3:
                    last_beep = now
                    threading.Thread(target=beep, daemon=True).start()
                key = cv2.waitKey(30) & 0xFF
            else:
                time.sleep(0.03)
            ended = cam is not None and cam.ended and cam.is_file
            if key in (ord("q"), 27) or ended or (args.seconds and now > args.seconds):
                break
            if key == ord("a"):
                spotter.acknowledge(now)
            elif key == ord("r"):
                spotter.reset()
            elif key == ord("s"):
                shared["scan"] = True
    except KeyboardInterrupt:
        pass
    finally:
        shared["stop"] = True
        if server is not None:
            server.shutdown()
        if writer is not None:
            writer.release()
            print(f"video  -> {out_dir / 'live.mp4'}")
        elif show_window:
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
