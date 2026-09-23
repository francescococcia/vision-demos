"""Segment a boulder problem with SAM 3.1, pose the climber with ViTPose, and
read the route off the pair.

    conda activate rock_climbing
    python main.py

Both models run on the VLM Run Gateway, so there are no weights to download.

With ``BATCH_MODE`` on, every clip in ``INPUT_DIR`` is read as another attempt at
the same route and each render ends on a panel comparing its sequence of holds
against the others'. The pipeline therefore runs in two passes: every clip is
analyzed first, because no clip can be rendered until all of them are known.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import (climb, compare, floor as floor_mod, holds as holds_mod, pose,
                 render, timing, video)
from src.env import load_api_key

console = Console()


def rule(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")


def kv(pairs: dict, title: str | None = None) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in pairs.items():
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, title_align="left", expand=False) if title else table)


def rel(path: Path) -> Path:
    """Project-relative path, for console output."""
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def route_metadata(prompt: str) -> dict:
    """What problem this is. Recorded into every artifact the run writes."""
    return {"color": cfg.HOLD_COLOR, "grade": cfg.ROUTE_GRADE,
            "name": cfg.ROUTE_NAME, "prompt": prompt}


def discover() -> list[Path]:
    """The clips to process, in a stable order.

    Batch mode takes the whole of INPUT_DIR — every clip in it is another
    attempt at the same route, which is the assumption the comparison rests on.
    Sorted by name so the reference clip and the panel's reading order do not
    move between runs.
    """
    if not cfg.BATCH_MODE:
        return [cfg.INPUT_VIDEO]
    if not cfg.INPUT_DIR.is_dir():
        return []
    suffixes = {s.lower() for s in cfg.INPUT_SUFFIXES}
    return sorted(p for p in cfg.INPUT_DIR.iterdir()
                  if p.suffix.lower() in suffixes and not p.name.startswith("."))


# ── one clip ─────────────────────────────────────────────────────────────────

@dataclass
class Run:
    """Everything one clip produced, up to but not including its render.

    Held whole because in batch mode the render cannot start until every clip
    has been analyzed: the panel a clip ends on is made of the other clips.
    """

    video: Path
    label: str
    run_dir: Path
    src_info: dict
    mp4: Path
    info: object
    export_mp4: Path
    export_info: object
    prompt: str
    hold_settings: dict
    route: list
    numbering: str
    ground: object
    poses: object
    analysis: object
    util: object
    rows: list
    warmup: int
    window: tuple
    extra_body: dict
    track_id: object
    track_stats: dict
    request_timing: object
    pose_usage: dict | None
    holds_cached: bool
    poses_cached: bool
    tonemap: str | None
    tonemap_reason: str
    convert_seconds: float = 0.0
    convert_cached: bool = True
    hold_seconds: float = 0.0
    hold_cost: float = 0.0
    floor_cost: float = 0.0
    upload_mb: float = 0.0
    started: float = 0.0      # perf counter at the top of `analyze`, so the
                              # per-clip metrics cover the analysis too

    @property
    def aspect(self) -> float:
        return self.info.width / self.info.height

    def attempt(self) -> compare.Attempt:
        """This clip as a row in the comparison."""
        return compare.Attempt(
            label=self.label,
            video=self.video.name,
            holds=self.route,
            sequence=climb.activation_sequence(self.analysis),
            limb_sequence=climb.limb_sequence(self.analysis),
            elapsed=(round(self.analysis.elapsed, 3)
                     if self.analysis.elapsed is not None else None),
            topped_out=self.analysis.completion_frame is not None,
            n_detected=len(self.route),
            n_used=self.analysis.n_used)


def analyze(source: Path, *, api_key: str, run_dir: Path, batch: bool) -> Run | None:
    """Steps 2-6 for one clip: convert, segment, pose, and read the climb.

    Everything up to the render, which in batch mode has to wait for the others.
    """
    from openai import OpenAI

    t_run = time.perf_counter()
    label = source.stem

    # ── 2. Source ────────────────────────────────────────────────────────────
    rule(2, f"Source video{f' — [bold]{label}[/]' if batch else ''}")
    if not source.is_file():
        console.print(f"[red]Not found:[/] {source}")
        return None

    src_info = video.probe_source(source)
    kv({
        "path": rel(source),
        "size": f"{source.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "rotation": f"{src_info['rotation']}°"
                    + ("  [dim](applied on convert; OpenCV would ignore it)[/]"
                       if src_info["rotation"] else ""),
        "colour": f"{src_info['color_primaries'] or '?'} / {src_info['color_transfer'] or '?'}"
                  + ("  [yellow](HDR)[/]" if src_info["is_hdr"] else "  [dim](SDR)[/]"),
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
    })

    # ── 3. Convert (cached) ──────────────────────────────────────────────────
    rule(3, "Convert to MP4")
    # Decided before the cache key is built, so switching tone-mapping on or off
    # produces a different MP4 rather than silently reusing the other one.
    tonemap, tonemap_reason = video.tonemap_filter(
        src_info, mode=cfg.TONEMAP, algorithm=cfg.TONEMAP_ALGORITHM,
    )
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{tonemap_reason}")

    def prepare(height, tag):
        """Convert to *height* (None = source), reusing a cached copy if present."""
        path = video.cache_path(source, cfg.CACHE_DIR, target_height=height,
                                trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                                tonemap=tonemap)
        seconds, cached = 0.0, True
        if path.is_file() and not cfg.FORCE_RECONVERT:
            console.print(f"  [green]cache hit[/] ({tag}): [dim]{rel(path)}[/]")
        else:
            cached = False
            reason = "FORCE_RECONVERT" if path.is_file() else "no cached copy"
            console.print(f"  [yellow]converting[/] ({tag}, {reason})")
            t0 = time.perf_counter()
            video.convert(source, path, target_height=height,
                          trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                          total_frames=src_info["n_frames"], console=console,
                          tonemap=tonemap)
            seconds = time.perf_counter() - t0
            console.print(f"  done in {seconds:.1f}s -> [dim]{rel(path)}[/]")
        return path, video.inspect(path), seconds, cached

    # Two renditions, because they answer to different limits: the models cap at
    # a 2048px long edge internally, so uploading more is pure wait, while the
    # export answers only to what you want to watch. Every coordinate on both
    # sides is normalized, so one inference serves any render size.
    mp4, info, convert_seconds, convert_cached = prepare(cfg.INFERENCE_HEIGHT, "inference")

    if cfg.EXPORT_HEIGHT == cfg.INFERENCE_HEIGHT:
        export_mp4, export_info = mp4, info
    else:
        export_mp4, export_info, extra_s, extra_cached = prepare(cfg.EXPORT_HEIGHT, "export")
        convert_seconds += extra_s
        convert_cached = convert_cached and extra_cached

    console.print(f"  inference [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")
    if export_mp4 is not mp4:
        console.print(f"  export    [bold]{export_info}[/]")

    # ── 4. Holds ─────────────────────────────────────────────────────────────
    rule(4, "The route (SAM 3.1)")
    prompt = cfg.HOLD_PROMPT.format(color=cfg.HOLD_COLOR)
    hold_settings = {
        "sample_frames": cfg.HOLD_SAMPLE_FRAMES, "min_score": cfg.HOLD_MIN_SCORE,
        "iou": cfg.HOLD_IOU_THRESHOLD, "min_appearance": cfg.HOLD_MIN_APPEARANCE,
        "raster": cfg.HOLD_RASTER_SIZE, "vote": cfg.HOLD_VOTE_FRACTION,
        "min_area": cfg.HOLD_MIN_AREA_PX, "fill_holes": cfg.HOLD_FILL_HOLES,
    }
    holds_cache = holds_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.HOLD_MODEL,
                                       prompt=prompt, settings=hold_settings)
    cached_holds = holds_mod.load_cache(holds_cache) if cfg.REUSE_HOLDS else None

    kv({"model": cfg.HOLD_MODEL, "prompt": f"[bold]{prompt}[/]",
        "sampled frames": cfg.HOLD_SAMPLE_FRAMES}, title="request")

    hold_seconds = 0.0
    if cached_holds is not None:
        route, raw_detections, hold_usages, created = cached_holds
        console.print(f"  [green]cache hit[/]: holds from [dim]{created}[/] "
                      f"([dim]{holds_cache.name}[/]); no gateway call")
        console.print("  [dim]set REUSE_HOLDS = False in config.py to re-segment[/]")
    else:
        if cfg.REUSE_HOLDS:
            console.print("  [yellow]no cached holds for these settings[/], calling the gateway")
        client = OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                        timeout=cfg.REQUEST_TIMEOUT, max_retries=1)
        frames = holds_mod.sample_frames(mp4, cfg.HOLD_SAMPLE_FRAMES)
        t0 = time.perf_counter()
        raw_detections, hold_usages = holds_mod.segment_frames(
            client, frames, model=cfg.HOLD_MODEL, prompt=prompt,
            min_score=cfg.HOLD_MIN_SCORE, workers=cfg.HOLD_REQUEST_WORKERS,
            console=console)
        hold_seconds = time.perf_counter() - t0
        route = holds_mod.consensus(
            raw_detections, iou_threshold=cfg.HOLD_IOU_THRESHOLD,
            min_appearance=cfg.HOLD_MIN_APPEARANCE, raster_size=cfg.HOLD_RASTER_SIZE,
            vote_fraction=cfg.HOLD_VOTE_FRACTION, min_area_px=cfg.HOLD_MIN_AREA_PX,
            fill_holes=cfg.HOLD_FILL_HOLES)
        holds_mod.save_cache(holds_cache, route, raw_detections, hold_usages,
                             stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        console.print(f"  [green]done in {hold_seconds:.1f}s[/]")

    # After the cache, not before it: a suppression is a decision about the
    # holds SAM already returned, so re-tuning it costs nothing and an existing
    # cache stays valid.
    n_clustered = len(route)
    route, swallowed = holds_mod.suppress_contained(
        route, max_containment=cfg.HOLD_NMS_CONTAINMENT)
    for small, big, share in swallowed:
        console.print(f"  [yellow]dropped a hold[/] at "
                      f"{tuple(round(v, 3) for v in small['bbox'])} "
                      f"(score {small['score']:.2f}) — [dim]{share:.0%} of it sits "
                      f"inside a larger hold scoring {big['score']:.2f}; "
                      f"HOLD_NMS_CONTAINMENT = {cfg.HOLD_NMS_CONTAINMENT}[/]")

    per_call = [len(d.items) for d in raw_detections]
    hold_cost = sum(u.get("cost") or 0.0 for u in hold_usages)
    kv({
        "instances per frame": f"{min(per_call)}-{max(per_call)} "
                               f"(mean {sum(per_call) / len(per_call):.1f})" if per_call else "—",
        "holds after consensus": f"[bold]{len(route)}[/] "
                                 f"(IoU ≥ {cfg.HOLD_IOU_THRESHOLD}, seen in "
                                 f"≥ {cfg.HOLD_MIN_APPEARANCE:.0%} of samples"
                                 + (f"; {n_clustered - len(route)} swallowed by a "
                                    f"larger hold)" if swallowed else ")"),
        "cost": f"${hold_cost:.4f}" if hold_cost else "—",
    }, title="response")

    if not route:
        console.print(f"[red]No holds found for {prompt!r}.[/] Try another HOLD_COLOR, "
                      "or lower HOLD_MIN_APPEARANCE in config.py.")
        return None

    # ── 4b. The floor ────────────────────────────────────────────────────────
    ground = None
    floor_cost = 0.0
    if cfg.DETECT_FLOOR:
        floor_settings = {"sample_frames": cfg.HOLD_SAMPLE_FRAMES,
                          "min_score": cfg.FLOOR_MIN_SCORE,
                          "min_area": cfg.FLOOR_MIN_AREA,
                          "resolution": cfg.FLOOR_EDGE_RESOLUTION}
        floor_cache = floor_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.HOLD_MODEL,
                                           prompt=cfg.FLOOR_PROMPT,
                                           settings=floor_settings)
        cached_floor = floor_mod.load_cache(floor_cache) if cfg.REUSE_FLOOR else None
        if cached_floor is not None:
            ground, floor_usages, created = cached_floor
            console.print(f"  floor: [green]cache hit[/] from [dim]{created}[/]")
        else:
            client = OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                            timeout=cfg.REQUEST_TIMEOUT, max_retries=1)
            sampled = holds_mod.sample_frames(mp4, cfg.HOLD_SAMPLE_FRAMES)
            ground, floor_usages = floor_mod.segment(
                client, sampled, model=cfg.HOLD_MODEL, prompt=cfg.FLOOR_PROMPT,
                resolution=cfg.FLOOR_EDGE_RESOLUTION, min_score=cfg.FLOOR_MIN_SCORE,
                min_area=cfg.FLOOR_MIN_AREA, workers=cfg.HOLD_REQUEST_WORKERS,
                console=console)
            floor_mod.save_cache(floor_cache, ground, floor_usages,
                                 stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        floor_cost = sum(u.get("cost") or 0.0 for u in floor_usages)
        if ground is not None:
            lo, hi = float(np.nanmin(ground.edge)), float(np.nanmax(ground.edge))
            console.print(f"  floor line at y [bold]{lo:.3f}-{hi:.3f}[/] "
                          f"([dim]{cfg.FLOOR_PROMPT!r}; slopes "
                          f"{(hi - lo) * export_info.height:.0f}px across frame[/])")
        else:
            console.print(f"  [yellow]no floor found[/] for {cfg.FLOOR_PROMPT!r}; "
                          "the start rule falls back to both feet on holds")

    # ── 5. Pose ──────────────────────────────────────────────────────────────
    rule(5, "The climber (ViTPose)")
    video_b64, extra_body = pose.build_request(
        mp4, every_frame=cfg.EVERY_FRAME, fps=info.fps, n_frames=info.n_frames,
        video_fps=cfg.VIDEO_FPS, video_max_frames=cfg.VIDEO_MAX_FRAMES,
        precision=cfg.PRECISION)
    upload_mb = len(video_b64) / 1e6
    kv({
        "model": cfg.POSE_MODEL,
        "mode": ("every frame, detector stride 1, nothing skipped"
                 if cfg.EVERY_FRAME else f"video_fps={cfg.VIDEO_FPS} (detector cadence)"),
        "billed units": f"~{extra_body.get('video_max_frames') or min(info.n_frames, 900)} frames",
        "upload": f"{upload_mb:.1f} MB base64",
    }, title="request")

    poses_cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.POSE_MODEL,
                                  extra_body=extra_body)
    cached_poses = pose.load_cache(poses_cache) if cfg.REUSE_POSES else None

    if cached_poses is not None:
        payload, pose_usage, cached_timing, created = cached_poses
        request_timing = timing.RequestTiming(**cached_timing) if cached_timing \
            else timing.RequestTiming()
        elapsed = request_timing.round_trip
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/] "
                      f"([dim]{poses_cache.name}[/]); no gateway call")
        console.print("  [dim]set REUSE_POSES = False in config.py to re-run detection[/]")
    else:
        if cfg.REUSE_POSES:
            console.print("  [yellow]no cached poses for these settings[/], calling the gateway")
        marks: dict = {}
        http_client = timing.timed_http_client(cfg.REQUEST_TIMEOUT, marks)
        client = OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                        timeout=cfg.REQUEST_TIMEOUT, max_retries=1,
                        **({"http_client": http_client} if http_client else {}))
        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/]; pose runs on every decoded frame…",
                            spinner="dots"):
            payload, pose_usage = pose.request_poses(
                client, model=cfg.POSE_MODEL, video_b64=video_b64, extra_body=extra_body)
        elapsed = time.perf_counter() - t0
        request_timing = timing.summarize(marks, elapsed)
        pose.save_cache(poses_cache, payload, pose_usage, asdict(request_timing),
                        stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if request_timing.measured_split:
            console.print(f"  [green]done in {elapsed:.1f}s[/]: "
                          f"upload {request_timing.upload:.1f}s, "
                          f"[bold]server {request_timing.server:.1f}s[/], "
                          f"download {request_timing.download:.1f}s")
        else:
            console.print(f"  [green]done in {elapsed:.1f}s[/]")
    del video_b64   # tens of MB per clip, and a batch holds every clip at once

    items, returned_frames = pose.unwrap(payload)

    # Which body is the climber, decided once against the route's own footprint.
    wall = holds_mod.wall_mask(route, res=cfg.WALL_MASK_RES, dilate=cfg.WALL_MASK_DILATE)
    track_id, track_stats = pose.pick_track(
        items, mask=wall, overlap_fn=holds_mod.wall_overlap,
        min_overlap=cfg.MIN_WALL_OVERLAP, lock_to_wall=cfg.LOCK_TO_WALL)
    poses = pose.resolve(items, returned_frames, track_id,
                         min_kpt_score=cfg.POSE_MIN_KPT_SCORE)
    poses = pose.smooth(poses, sigma=cfg.POSE_SMOOTH_SIGMA)

    coverage = poses.n_frames_returned / info.n_frames if info.n_frames else 0
    kv({
        "frames returned": f"{poses.n_frames_returned} of {info.n_frames} source ({coverage:.0%})",
        "tracks seen": ", ".join(
            f"{t}{' [bold](climber)[/]' if t == track_id else ''} "
            f"[dim]({s['coverage']:.0%} of frames, {s['mean_overlap']:.2f} on-wall)[/]"
            for t, s in sorted(track_stats.items(), key=lambda kv: -kv[1]["coverage"])) or "none",
        "frames with the climber": poses.n_posed,
        "joint confidence": (f"≥ {cfg.POSE_MIN_KPT_SCORE}" if cfg.POSE_MIN_KPT_SCORE
                             else "off (the (0, 0) sentinel alone)"),
        "smoothing": f"σ = {cfg.POSE_SMOOTH_SIGMA} frames" if cfg.POSE_SMOOTH_SIGMA else "off",
        "usage": pose_usage or "—",
    }, title="response")

    if not poses.n_posed:
        console.print("[red]No climber found in any frame.[/]")
        return None

    # Written now rather than with the rest of the artifacts, so the raw
    # response can be dropped before the next clip is loaded: a batch otherwise
    # holds every clip's frame-by-frame keypoints in memory at once.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "poses.json").write_text(json.dumps(payload))
    del payload, items

    # The colour prompt finds every hold of that colour in shot, which is more
    # wall than this route. The body's own trajectory is what narrows it.
    if cfg.HOLD_SPATIAL_FILTER:
        before = len(route)
        route, dropped = holds_mod.filter_by_pose_region(
            route, pose.confident_points(poses), margin=cfg.HOLD_SPATIAL_MARGIN)
        if dropped:
            console.print(f"  [dim]{dropped} of {before} holds dropped: outside the "
                          f"climber's own reach (HOLD_SPATIAL_MARGIN = "
                          f"{cfg.HOLD_SPATIAL_MARGIN})[/]")
    route, numbering = holds_mod.assign_ids(
        route, band=cfg.HOLD_NUMBER_BAND, direction=cfg.HOLD_NUMBERING)
    console.print(f"  [dim]numbered bottom to top, "
                  f"{'left to right' if numbering == 'ltr' else 'right to left'} "
                  f"within a row (lean {holds_mod.route_lean(route):+.2f}"
                  f"{', forced' if cfg.HOLD_NUMBERING != 'auto' else ''})[/]")

    # ── 6. The climb ─────────────────────────────────────────────────────────
    rule(6, "The climb")
    analysis = climb.analyze(route, poses, fps=info.fps,
                             aspect=info.width / info.height, cfg=cfg, floor=ground)
    # The utilization window is the climb itself, so warm-up touches before the
    # clock started are not counted into the split.
    frames = poses.frames()
    window = (analysis.start_frame if analysis.start_frame is not None else frames[0],
              analysis.completion_frame if analysis.completion_frame is not None
              else frames[-1])
    # One window governs the panel, the pips and every reported figure, so they
    # cannot disagree about whether a warm-up touch counts.
    warmup = climb.restrict(analysis, window)
    util = climb.Utilization(analysis, window=window)
    rows = climb.hold_times(analysis, cfg=cfg)

    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("holds on route", f"[bold]{len(route)}[/]")
    table.add_row("holds used", f"[bold]{analysis.n_used}[/] of {len(route)}")
    table.add_row("start", f"frame {analysis.start_frame} "
                           f"({analysis.start_frame / info.fps:.2f}s)"
                  if analysis.start_frame is not None else "[yellow]not detected[/]")
    table.add_row("topped out", f"frame {analysis.completion_frame} "
                                f"(hold {analysis.final_hold_id})"
                  if analysis.completion_frame is not None else "[yellow]no[/]")
    if analysis.elapsed is not None:
        table.add_row("time on the wall", f"[bold green]{analysis.elapsed:.2f}s[/]")
    table.add_row("sequence", "[bold]"
                  + " ".join(str(h) for h in climb.activation_sequence(analysis))
                  + "[/]" if analysis.n_used else "—")
    console.print(Panel(table, title="[bold]climb[/]", title_align="left", expand=False))

    limb_table = Table.grid(padding=(0, 3))
    limb_table.add_column(style="dim", justify="right")
    limb_table.add_column(justify="right")
    limb_table.add_column(justify="right")
    limb_table.add_column(justify="right")
    limb_table.add_column()
    limb_table.add_row("", "contact", "on hold", "holds", "in order (the beta)")
    for limb in climb.LIMBS:
        limb_table.add_row(
            limb.name,
            f"{util.seconds(limb.key):.2f}s",
            f"[bold]{util.coverage(limb.key) * 100:.0f}%[/]",
            f"{len(util.holds_by_limb[limb.key])}",
            " ".join(str(h) for h in util.sequence(limb.key)) or "—")
    console.print(Panel(
        limb_table,
        title="[bold]limb utilization[/]  [dim]on hold = this limb's contact time "
              "over the climb; the four do not add to 100%[/]",
        title_align="left", expand=False))

    for problem in climb.check(analysis, util):
        console.print(f"  [red]inconsistent:[/] {problem}")

    if analysis.start_frame is None:
        console.print("  [yellow]warning[/] no start detected: both feet never dwelt on a "
                      "hold, so the clock and the route line are off. Lower "
                      "HOLD_DWELL_SECONDS or raise ANKLE_TO_TOE_OFFSET.")
    if analysis.completion_frame is None:
        console.print("  [yellow]warning[/] no top-out: both wrists never dwelt on hold "
                      f"{analysis.final_hold_id} together. Lower "
                      "FINAL_HOLD_DWELL_SECONDS, or check that the top hold was found.")

    return Run(
        video=source, label=label, run_dir=run_dir, src_info=src_info, mp4=mp4,
        info=info, export_mp4=export_mp4, export_info=export_info, prompt=prompt,
        hold_settings=hold_settings, route=route, numbering=numbering, ground=ground,
        poses=poses, analysis=analysis, util=util, rows=rows, warmup=warmup,
        window=window, extra_body=extra_body, track_id=track_id,
        track_stats=track_stats, request_timing=request_timing, pose_usage=pose_usage,
        holds_cached=cached_holds is not None, poses_cached=cached_poses is not None,
        tonemap=tonemap, tonemap_reason=tonemap_reason,
        convert_seconds=convert_seconds, convert_cached=convert_cached,
        hold_seconds=hold_seconds, hold_cost=hold_cost, floor_cost=floor_cost,
        upload_mb=upload_mb, started=t_run)


# ── the comparison ───────────────────────────────────────────────────────────

def compare_runs(runs: list[Run]) -> compare.Comparison | None:
    """Put every attempt into one numbering, and say where the walls disagree.

    This is the step that makes "1 2 5 8" from one clip mean the same thing as
    "1 2 5 8" from another. Without it the numbers are four private languages
    that happen to share an alphabet.
    """
    if len(runs) < 2:
        return None

    rule(7, "The same route, four ways")
    comparison = compare.build(
        [run.attempt() for run in runs],
        aspect=runs[0].aspect,
        max_distance=cfg.HOLD_MATCH_DISTANCE,
        drift_warn=cfg.HOLD_DRIFT_WARN,
        align_ids=cfg.ALIGN_HOLD_IDS,
        route=route_metadata(runs[0].prompt))

    reference_run = next(r for r in runs if r.label == comparison.reference)
    counts = sorted({len(run.route) for run in runs})
    console.print(f"  reference wall: [bold]{comparison.reference}[/] "
                  f"([dim]{len(reference_run.route)} holds — the count the clips "
                  f"agree on{'' if len(counts) == 1 else f'; the clips found {counts}'}; "
                  f"every other clip's holds are matched onto it by position[/])")

    table = Table.grid(padding=(0, 3))
    for column, justify in (("attempt", "right"), ("holds", "right"),
                            ("matched", "right"), ("time", "right"), ("sequence", "left")):
        table.add_column(justify=justify, style="dim" if column == "attempt" else None)
    table.add_row("", "found", "matched", "time", "sequence (reference numbering)")
    shortest = comparison.shortest()
    for attempt in comparison.attempts:
        stats = attempt.alignment
        sequence = " ".join(str(h) for h in attempt.aligned_sequence) or "—"
        tag = "  [bold yellow]shortest[/]" if shortest and \
            attempt.label == shortest.label else ""
        table.add_row(
            attempt.label,
            str(attempt.n_detected),
            f"{stats['matched']}/{stats['reference_holds']}",
            f"{attempt.elapsed:.2f}s" if attempt.elapsed is not None else "—",
            f"{sequence}{tag}")
    console.print(Panel(table, title="[bold]attempts[/]", title_align="left", expand=False))

    if cfg.CHECK_HOLD_CONSISTENCY:
        if comparison.problems:
            for problem in comparison.problems:
                console.print(f"  [yellow]hold check:[/] {problem}")
            if cfg.HOLD_CONSISTENCY_STRICT:
                console.print("  [red]HOLD_CONSISTENCY_STRICT is on[/] — stopping rather "
                              "than comparing sequences that may not line up")
                raise SystemExit(1)
            console.print("  [dim]sequences are compared in the reference clip's "
                          "numbering, so the matched holds still line up[/]")
        else:
            offsets = [a.alignment["max_offset"] for a in comparison.attempts
                       if a.alignment.get("max_offset") is not None]
            console.print(f"  [green]walls agree[/]: every clip found the same "
                          f"{comparison.attempts[0].n_detected} holds, within "
                          f"{max(offsets):.4f} of a frame height of each other")
    return comparison


# ── one clip's output ────────────────────────────────────────────────────────

def adopt_numbering(runs: list[Run], comparison: compare.Comparison) -> None:
    """Renumber every clip's holds into the run's one numbering.

    Up to here the alignment only reaches the comparison: each clip still draws
    its own SAM ids on the wall, so a clip whose wall segmented differently
    shows one number in the video and another on the card under it — the video
    says you matched hold 15, the card says 14. They are the same hold. Nothing
    but the numbering disagrees, and the numbering is ours to choose.

    So the clip adopts it. From here the ids in the overlay, holds.png,
    summary.txt, the CSVs and the JSON are all the reference clip's, and the
    only artifact that still speaks a clip's private numbering is its cache.
    """
    if not comparison.aligned:
        return

    renumbered = []
    for run in runs:
        attempt = comparison.by_label(run.label)
        if attempt is None or not attempt.numbering:
            continue
        # `run.route` is the same list of dicts as `run.analysis.holds`, so this
        # renumbers the wall and the climb together.
        climb.renumber(run.analysis, attempt.numbering)
        # Both read hold ids at construction and keep their own copies.
        run.util = climb.Utilization(run.analysis, window=run.window)
        run.rows = climb.hold_times(run.analysis, cfg=cfg)
        # The attempt's own-id sequence is now the aligned one; leaving it as it
        # was would put two numberings back into comparison.json.
        attempt.sequence = list(attempt.aligned_sequence)
        if attempt.alignment.get("renumbered") or attempt.alignment.get("unmatched_in_other"):
            renumbered.append(run.label)

    if renumbered:
        console.print(f"  [dim]{', '.join(renumbered)} renumbered onto "
                      f"{comparison.reference}'s wall — every artifact and the "
                      f"ids drawn on the video now use that numbering[/]")


def deliver(run: Run, comparison: compare.Comparison | None, *, step: int,
            batch: bool) -> Path:
    """Steps 8-10 for one clip: render it, then write everything it produced."""
    rule(step, f"Render{f' — [bold]{run.label}[/]' if batch else ''}")
    run_dir = run.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis, info, export_info = run.analysis, run.info, run.export_info

    # Written here rather than at the end of `analyze`: the ids are not settled
    # until every clip has been read and the numbering chosen.
    (run_dir / "holds.json").write_text(json.dumps(run.route, indent=2))

    if cfg.SAVE_HOLDS_PREVIEW:
        cap = cv2.VideoCapture(str(run.export_mp4))
        ok, first = cap.read()
        cap.release()
        if ok:
            preview = run_dir / "holds.png"
            color = cfg.HOLD_RENDER_COLOR.get(cfg.HOLD_COLOR, cfg.HOLD_COLOR_FALLBACK)
            cv2.imwrite(str(preview), render.holds_preview(
                first, run.route, color, scale=export_info.width / 608.0))
            console.print(f"  holds  -> [dim]{rel(preview)}[/]")

    route_points = climb.route_path(analysis, export_info.width, export_info.height,
                                    sigma=cfg.ROUTE_SPLINE_SIGMA)

    raw_pair = run_dir / "_pair.mp4"
    raw_solo = run_dir / "_solo.mp4" if cfg.SAVE_RIGHT_PANEL else None
    t0 = time.perf_counter()
    stats = render.render(export_info, run.route, run.poses, analysis, route_points,
                          raw_pair, cfg=cfg, console=console, right_path=raw_solo,
                          ground=run.ground, comparison=comparison, label=run.label)
    render_seconds = time.perf_counter() - t0
    console.print(f"  drew the climber on [bold]{stats['frames_with_pose']}/"
                  f"{stats['frames_written']}[/] frames")
    if stats["tail_frames"]:
        console.print(f"  [dim]completion panel held for {stats['tail_frames']} frames "
                      f"({stats['tail_frames'] / export_info.fps:.1f}s) past the end "
                      f"of the clip[/]")

    # The render is longer than its source once the panel is held, so the length
    # is stated rather than left to `-shortest`, which would trim the tail away.
    duration = (stats["frames_written"] / export_info.fps
                if stats["tail_frames"] else None)
    out_video = run_dir / f"{run.video.stem}_climb.mp4"
    t0 = time.perf_counter()
    audio = run.video if cfg.COPY_AUDIO else None
    video.encode_h264(raw_pair, out_video, crf=cfg.OUTPUT_CRF, audio_from=audio,
                      duration=duration)
    encode_seconds = time.perf_counter() - t0
    raw_pair.unlink(missing_ok=True)
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB)")

    solo_video = None
    if raw_solo is not None:
        solo_video = run_dir / f"{run.video.stem}_route.mp4"
        t0 = time.perf_counter()
        video.encode_h264(raw_solo, solo_video, crf=cfg.OUTPUT_CRF, audio_from=audio,
                          duration=duration)
        encode_seconds += time.perf_counter() - t0
        raw_solo.unlink(missing_ok=True)
        console.print(f"  route  -> [dim]{rel(solo_video)}[/]")

    # ── artifacts ────────────────────────────────────────────────────────────
    rule(step + 1, "Artifacts")
    util, rows = run.util, run.rows

    with (run_dir / "hold_times.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "hold_id", "order", "limbs", "time_start", "time_end", "contact_seconds"])
        writer.writeheader()
        writer.writerows(rows)

    # One row per (limb, hold) that actually happened — the join key for
    # comparing two climbers on the same route is (hold_id, limb).
    with (run_dir / "limb_usage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["limb", "hold_id", "seconds", "frames", "share_of_contact"])
        writer.writeheader()
        writer.writerows(util.rows())

    sequence = climb.activation_sequence(analysis)
    moves = climb.limb_sequence(analysis)
    attempt = comparison.by_label(run.label) if comparison else None

    (run_dir / "climb.json").write_text(json.dumps({
        "route": route_metadata(run.prompt),
        "route_color": cfg.HOLD_COLOR, "prompt": run.prompt, "fps": info.fps,
        "hold_numbering": {"order": "bottom_to_top", "within_row": run.numbering,
                           "band": cfg.HOLD_NUMBER_BAND,
                           "lean": round(holds_mod.route_lean(run.route), 4)},
        **analysis.summary(),
        "sequence": sequence,
        "moves": moves,
        "utilization": util.as_dict(),
        "holds": rows,
    }, indent=2))

    # The sequence, on its own and in full. Two of them, because they answer
    # different questions: `sequence` is the path up the wall and is what two
    # attempts are compared on, `moves` is which appendage went where and is
    # what tells you *how* the path was climbed.
    (run_dir / "sequence.json").write_text(json.dumps({
        "video": run.video.name,
        "label": run.label,
        "route": route_metadata(run.prompt),
        "topped_out": analysis.completion_frame is not None,
        "elapsed_seconds": round(analysis.elapsed, 3)
        if analysis.elapsed is not None else None,
        "holds_detected": len(run.route),
        "holds_used": analysis.n_used,
        "sequence": sequence,
        "sequence_aligned": attempt.aligned_sequence if attempt else sequence,
        "first_limb_by_hold": {str(k): v for k, v in
                               sorted(climb.first_limb_by_hold(analysis).items())},
        "moves": moves,
        "per_limb_sequence": {limb.key: util.sequence(limb.key) for limb in climb.LIMBS},
        "comparison": None if comparison is None else {
            "reference": comparison.reference,
            "ids_aligned": comparison.aligned,
            "shortest": comparison.shortest().label if comparison.shortest() else None,
            "fastest": comparison.fastest().label if comparison.fastest() else None,
            "shared_holds": sorted(comparison.shared_holds()),
            "others": [{"label": other.label,
                        "sequence": other.aligned_sequence,
                        "holds": len(other.aligned_sequence),
                        "elapsed_seconds": other.elapsed,
                        "topped_out": other.topped_out}
                       for other in comparison.others(run.label)],
        },
    }, indent=2))

    (run_dir / "summary.txt").write_text(climb.summary_text(
        analysis, rows, util, source=run.video.name, color=cfg.HOLD_COLOR,
        numbering=run.numbering, warmup=run.warmup))
    console.print(f"  holds.json, poses.json, climb.json, sequence.json, "
                  f"hold_times.csv, limb_usage.csv, summary.txt -> [dim]{rel(run_dir)}/[/]")

    # ── performance ──────────────────────────────────────────────────────────
    rule(step + 2, "Performance")
    total = time.perf_counter() - run.started
    pose_cost = (run.pose_usage or {}).get("cost")
    metrics = timing.Metrics(
        frames=run.poses.n_frames_returned, video_seconds=info.duration,
        convert_seconds=run.convert_seconds, convert_cached=run.convert_cached,
        encode_seconds=encode_seconds, request=run.request_timing,
        render_seconds=render_seconds, total_seconds=total,
        upload_mb=run.upload_mb, cost_usd=pose_cost,
        poses_cached=run.poses_cached)

    (run_dir / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2))
    (run_dir / "metrics.txt").write_text(metrics.as_text())

    m = metrics.as_dict()
    total_cost = (pose_cost or 0.0) + run.hold_cost + run.floor_cost
    kv({
        "ViTPose throughput": f"[bold green]{m['pose']['fps']:.1f} fps[/]  "
                              f"({m['pose']['ms_per_frame']:.0f} ms/frame, "
                              f"{m['pose']['realtime_factor']:.2f}x realtime)",
        "SAM 3.1": f"{run.hold_seconds:.1f}s for {cfg.HOLD_SAMPLE_FRAMES} frames"
                   if run.hold_seconds else "cached",
        "gateway round trip": f"{m['gateway_round_trip']['seconds']:.2f}s",
        "render": f"{m['local']['render_seconds']:.2f}s ({m['local']['render_fps']:.1f} fps)",
        "cost": f"${total_cost:.4f}  [dim](pose ${pose_cost or 0:.4f} + "
                f"holds ${run.hold_cost:.4f}"
                + (f" + floor ${run.floor_cost:.4f}" if run.floor_cost else "") + ")[/]"
                if total_cost else "—",
    }, title="metrics")

    (run_dir / "run.json").write_text(json.dumps({
        "stamp": run_dir.name,
        "total_seconds": round(total, 2),
        "batch_mode": cfg.BATCH_MODE,
        "route": route_metadata(run.prompt),
        "input": {"path": str(run.video), **run.src_info},
        "converted": {"path": str(run.mp4), "width": info.width, "height": info.height,
                      "fps": info.fps, "n_frames": info.n_frames},
        "tonemap": {"mode": cfg.TONEMAP, "algorithm": cfg.TONEMAP_ALGORITHM,
                    "filter": run.tonemap, "reason": run.tonemap_reason},
        "holds": {"model": cfg.HOLD_MODEL, "prompt": run.prompt,
                  "settings": run.hold_settings, "detected": len(run.route),
                  "from_cache": run.holds_cached, "cost_usd": run.hold_cost},
        "pose": {"model": cfg.POSE_MODEL, "extra_body": run.extra_body,
                 "min_kpt_score": cfg.POSE_MIN_KPT_SCORE,
                 "smooth_sigma": cfg.POSE_SMOOTH_SIGMA,
                 "track_id": run.track_id, "tracks": run.track_stats,
                 "from_cache": run.poses_cached, "usage": run.pose_usage},
        "climb": {**analysis.summary(), "numbering": run.numbering,
                  "sequence": sequence},
        "floor": {"prompt": cfg.FLOOR_PROMPT, "found": run.ground is not None,
                  "clearance": cfg.FLOOR_CLEARANCE, "cost_usd": run.floor_cost}
                 if cfg.DETECT_FLOOR else None,
        "utilization": util.as_dict(),
        "comparison": None if attempt is None else {
            "reference": comparison.reference,
            "alignment": attempt.alignment,
            "sequence_aligned": attempt.aligned_sequence},
        "render": {**stats, "width": export_info.width, "height": export_info.height},
        "metrics": metrics.as_dict(),
        "outputs": {"video": out_video.name,
                    "route": solo_video.name if solo_video else None,
                    "holds": "holds.json", "climb": "climb.json",
                    "sequence": "sequence.json",
                    "limb_usage": "limb_usage.csv",
                    "summary": "summary.txt", "metrics": "metrics.json"},
    }, indent=2))
    return run_dir


def to_render(runs: list[Run], console) -> list[tuple[int, Run]]:
    """The attempts to render, as (attempt number, run). Empty if none matched.

    RENDER_ONLY narrows the render, never the analysis: the card a clip ends on
    is built from every clip, so all of them are analyzed either way. The number
    is the one the card shows, because both come from this same list.
    """
    numbered = list(enumerate(runs, start=1))
    want = cfg.RENDER_ONLY
    if want is None:
        return numbered
    if isinstance(want, int):
        chosen = [pair for pair in numbered if pair[0] == want]
    else:
        stem = Path(want).stem.lower()
        chosen = [pair for pair in numbered if pair[1].label.lower() == stem]
    if not chosen:
        console.print(f"[red]RENDER_ONLY = {want!r} matches no attempt.[/] Available: "
                      + ", ".join(f"{i} = {r.label}" for i, r in numbered))
    return chosen


def main() -> int:
    t_start = time.perf_counter()
    console.print()
    console.rule("[bold]SAM 3.1 + ViTPose: rock climbing[/]", align="center")

    # ── 1. Key ───────────────────────────────────────────────────────────────
    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  loaded from [green]{source}[/] ([dim]…{api_key[-4:]}[/])")

    sources = discover()
    if not sources:
        where = cfg.INPUT_DIR if cfg.BATCH_MODE else cfg.INPUT_VIDEO
        console.print(f"[red]No input videos found:[/] {where}")
        return 1

    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    batch = cfg.BATCH_MODE and len(sources) > 1
    batch_dir = cfg.OUTPUT_DIR / stamp
    if batch:
        console.print(f"  [bold]batch mode[/]: {len(sources)} attempts at the "
                      f"[bold]{cfg.HOLD_COLOR}[/] route"
                      + (f" ({cfg.ROUTE_GRADE})" if cfg.ROUTE_GRADE else "")
                      + f" — [dim]{rel(cfg.INPUT_DIR)}/[/]")
        for path in sources:
            console.print(f"    [dim]{path.name}[/]")

    # ── every clip, before any of them is rendered ───────────────────────────
    runs: list[Run] = []
    for i, path in enumerate(sources, start=1):
        if batch:
            console.print()
            console.rule(f"[bold magenta]attempt {i}/{len(sources)}[/] {path.name}",
                         align="left")
        run_dir = batch_dir / path.stem if batch else batch_dir
        run = analyze(path, api_key=api_key, run_dir=run_dir, batch=batch)
        if run is None:
            if not batch:
                return 1
            console.print(f"  [yellow]skipping {path.name}[/]; it produced nothing "
                          "to compare")
            continue
        runs.append(run)

    if not runs:
        console.print("[red]Nothing analyzed.[/]")
        return 1

    # ── the comparison, then the renders it feeds ────────────────────────────
    console.print()
    comparison = compare_runs(runs)
    if comparison is not None:
        adopt_numbering(runs, comparison)
    step = 8 if comparison is not None else 7

    selected = to_render(runs, console)
    if not selected:
        return 1
    if cfg.RENDER_ONLY is not None:
        only = ", ".join(f"attempt {i} ({r.label})" for i, r in selected)
        console.print(f"  [bold]rendering {only}[/] only; the rest were analyzed, "
                      "not drawn")

    for i, run in selected:
        if batch:
            console.print()
            console.rule(f"[bold magenta]{i}/{len(runs)}[/] {run.label}", align="left")
        deliver(run, comparison, step=step, batch=batch)

    if comparison is not None:
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "comparison.json").write_text(
            json.dumps(comparison.as_dict(), indent=2))
        (batch_dir / "sequences.txt").write_text(compare.summary_text(comparison))
        console.print()
        console.print(f"  comparison.json, sequences.txt -> [dim]{rel(batch_dir)}/[/]")

    total = time.perf_counter() - t_start
    console.print()
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(batch_dir)}/[/]\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        raise
