"""Listening gestures: pre-rendered clip specs and the per-session idle "director" (pure Python, unit-tested).

Clips are re-renders of a stretch of the avatar's idle loop with small extra controls on top (head pitch/yaw/roll
offsets, expression offsets). Every control is zero on the first and last frame, so a clip begins and ends on
exactly the idle frame it was cut from (see `DittoRenderer.render_idle`). The director plays the idle loop and
splices a clip in when playback passes a frame that matches the clip's first frame, then continues the idle loop
from the frame the clip ends on.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

FPS = 25

# Expression offsets on Ditto/LivePortrait keypoints (exp is 21 x 3), the same edits as the LivePortrait expression
# editor in ComfyUI-AdvancedLivePortrait (`calc_fe`); its pupil constants are identical to Ditto's `_eye_delta`.
_SMILE = {(20, 1): -0.01, (14, 1): -0.02, (17, 1): 0.0065, (17, 2): 0.003,
          (13, 1): -0.00275, (16, 1): -0.00275, (3, 1): -0.0035, (7, 1): -0.0035}
_PUPIL_Y = {(11, 1): -0.001, (15, 1): -0.001}
_JAW = {(19, 1): 0.001, (19, 2): 0.0001, (17, 1): -0.0001}     # the editor's "mouth" (jaw open)
_LIDS = {(11, 1): -0.001, (13, 1): 0.0003, (15, 1): -0.001, (16, 1): 0.0003, (1, 1): -0.00025, (2, 1): 0.00025}


def exp_offset(smile: float = 0.0, pupil_y: float = 0.0, lids: float = 0.0, jaw: float = 0.0) -> np.ndarray:
    """(1, 63) expression offset. smile ~0..1; pupil_y < 0 looks down; lids < 0 lowers the eyelids;
    jaw < 0 closes the mouth (a smile alone parts the lips)."""
    e = np.zeros((21, 3), np.float32)
    for table, amount in ((_SMILE, smile), (_PUPIL_Y, pupil_y), (_LIDS, lids), (_JAW, jaw)):
        for (k, c), v in table.items():
            e[k, c] += v * amount
    return e.reshape(1, 63).astype(np.float32)


def bump(n: int) -> list[float]:
    """0 -> 1 -> 0 raised cosine with zero slope at both ends."""
    if n < 3:
        return [0.0] * n
    return [float(v) for v in 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))]


def plateau(n: int, ramp: int) -> list[float]:
    """Smooth rise over `ramp` frames, hold, smooth fall; exactly 0 at both ends."""
    ramp = max(1, min(ramp, (n - 1) // 2))
    x = np.ones(n)
    r = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp + 1) / ramp)
    x[: ramp + 1] = r
    x[n - ramp - 1:] = r[::-1]
    return [float(v) for v in x]


# Signs and amplitudes, tuned by eye on Ditto (see README "Listening gestures").
# Ditto pitch: positive = head down. A smile parts the lips (teeth show from ~0.6); closing the jaw keeps them shut.
NOD_PITCH = 6.0           # degrees at the bottom of the nod
GLANCE_PITCH = 5.0
GLANCE_PUPIL = -9.0       # eyes down
GLANCE_LIDS = -4.0        # lids follow the eyes down
TILT_ROLL = 6.0
TILT_YAW = -3.0
SMILE = 0.8
SMILE_JAW = -12.0


def clip_controls(name: str, scale: float = 1.0) -> list[dict]:
    """Per-frame controls for a named clip (keys understood by ditto_engine.patch_motion_stitch)."""
    scale = float(scale)
    if name == "nod":                                   # "mm-hm": one soft dip, 0.8 s
        n = 20
        e = bump(n)
        return [{"add_pitch": NOD_PITCH * scale * v} for v in e]
    if name == "glance_down":                           # look at notes, 1.6 s
        n = 40
        e = plateau(n, 10)
        return [{"add_pitch": GLANCE_PITCH * scale * v,
                 "add_exp": exp_offset(pupil_y=GLANCE_PUPIL * scale * v, lids=GLANCE_LIDS * scale * v)} for v in e]
    if name == "tilt":                                  # slight head tilt ("hmm, tell me more"), 1.4 s
        n = 35
        e = plateau(n, 11)
        return [{"add_roll": TILT_ROLL * scale * v, "add_yaw": TILT_YAW * scale * v} for v in e]
    if name == "smile":                                 # small warm closed-mouth smile, 2 s
        n = 50
        e = plateau(n, 12)
        return [{"add_exp": exp_offset(smile=SMILE * scale * v, jaw=SMILE_JAW * scale * v), "add_pitch": 1.0 * scale * v} for v in e]
    raise KeyError(name)


CLIP_NAMES = ("nod", "glance_down", "tilt", "smile")


# ---------------------------------------------------------------------------------------------------------------
# Splice points
# ---------------------------------------------------------------------------------------------------------------
T_WEIGHT = 30.0       # head translation (Ditto `t`, ~face-size units) -> "degrees" for the pose distance
BLINK_COST = 100.0    # never splice between a blinking and an open-eyed frame


def pose_cost(poses: np.ndarray, blink: np.ndarray, anchors: list[int]) -> np.ndarray:
    """(len(anchors), N) distance from each idle frame to each anchor frame: |d pitch|+|d yaw|+|d roll| in degrees,
    plus translation, plus a large penalty if exactly one of the two frames is mid-blink.
    `poses` is (N, 5): pitch, yaw, roll (degrees), tx, ty."""
    p = np.asarray(poses, np.float64)
    w = np.array([1, 1, 1, T_WEIGHT, T_WEIGHT])
    b = np.asarray(blink, bool)
    return np.stack([(np.abs(p - p[a]) * w).sum(1) + BLINK_COST * (b != b[a]) for a in anchors])


def pick_anchors(poses: np.ndarray, blink: np.ndarray, k: int, clip_len: int) -> list[int]:
    """Greedy k-center-ish choice of splice frames: each new anchor most reduces the mean distance from any idle
    frame to its nearest anchor. Anchors leave room for the longest clip and are not mid-blink."""
    n = len(poses)
    cand = [a for a in range(max(1, n - clip_len)) if not blink[a]] or [0]
    full = pose_cost(poses, blink, cand)
    chosen: list[int] = []
    best = np.full(n, np.inf)
    for _ in range(max(1, min(k, len(cand)))):
        i = int(np.argmin([np.minimum(best, full[j]).mean() for j in range(len(cand))]))
        chosen.append(cand[i])
        best = np.minimum(best, full[i])
    return chosen


@dataclass
class IdleAssets:
    """Everything idle playback needs for one avatar (pre-rendered at startup, cached on the data volume)."""
    frames: list                                         # idle loop, played ping-pong
    clips: dict = field(default_factory=dict)            # name -> [frames rendered from anchor i] (same order as anchors)
    anchors: list = field(default_factory=list)          # idle frame index each clip variant starts on
    cost: Optional[np.ndarray] = None                    # (len(anchors), len(frames)) splice distance

    def clip_len(self, name: str) -> int:
        return len(self.clips[name][0])


# ---------------------------------------------------------------------------------------------------------------
# States -> clips
# ---------------------------------------------------------------------------------------------------------------
STATES = ("listening", "backchannel", "probe", "hint", "wrap_up", "silent_thinking", "paused")
# one-shot states: play one clip (first available in the list), the persistent mode is unchanged
ONE_SHOT = {"backchannel": ("nod",), "probe": ("tilt",), "hint": ("smile", "nod"), "wrap_up": ("smile", "nod")}
# persistent modes: occasional random gestures while idle, (clips, min s, max s between them)
MODES = {"listening": (("nod", "glance_down", "nod"), None, None),     # None: AVATAR_LISTEN_GESTURE_MIN/MAX_S
         "silent_thinking": (("glance_down",), 10.0, 20.0),
         "paused": ((), 0, 0)}


def ping_pong(pos: int, n: int) -> int:
    if n <= 1:
        return 0
    k = pos % (2 * n - 2)
    return k if k < n else 2 * n - 2 - k


class Director:
    """Per-session idle playback: the ping-pong idle loop, plus gesture clips spliced in on request.

    `frame(tick)` is called by the video track for every tick on which no utterance is showing (same thread as
    `set_state`, the asyncio loop). Clips never start while `busy()` (an utterance is queued or playing); a
    playing clip is simply cut when the utterance's first frame is due, so speech is never delayed.
    """

    XFADE = 4                     # frames of cross-fade into a clip (hides the residual pose difference)
    THR0, THR_SLOPE = 0.6, 2.0    # splice when distance <= THR0 + THR_SLOPE * seconds waited

    def __init__(self, assets: Optional[IdleAssets], *, busy: Callable[[], bool] = lambda: False,
                 on_event: Callable[[dict], None] = lambda e: None, rng: Optional[random.Random] = None,
                 listen_interval: tuple[float, float] = (6.0, 15.0), queue_s: float = 8.0):
        self.a = assets
        self.busy, self.on_event = busy, on_event
        self.rng = rng or random.Random()
        self.listen_interval, self.queue_s = listen_interval, queue_s
        self.mode = "idle"            # no random gestures until the client says otherwise
        self.pos = 0                  # ping-pong position in the idle loop
        self.last_tick: Optional[int] = None
        self.clip: Optional[dict] = None
        self.pending: Optional[dict] = None
        self.next_auto: Optional[int] = None
        self._auto_clips: list[str] = []
        self.tick = 0

    # ---- control (API) ----
    def available(self, name: str) -> bool:
        return bool(self.a and name in self.a.clips and self.a.anchors)

    def set_state(self, state: str) -> dict:
        if state not in STATES:
            raise ValueError(state)
        if state in ONE_SHOT:
            clip = next((c for c in ONE_SHOT[state] if self.available(c)), None)
            if clip is None:
                self.on_event({"type": "state_applied", "state": state, "clip": None})
                return {"state": state, "mode": self.mode, "clip": None, "status": "unavailable"}
            self.pending = {"clip": clip, "state": state, "at": self.tick, "wait_from": None}
            return {"state": state, "mode": self.mode, "clip": clip, "status": "queued" if self.busy() else "pending"}
        self.mode = state
        self.next_auto = None
        if state == "paused":
            self.pending = None
        self.on_event({"type": "state_applied", "state": state, "clip": None})
        return {"state": state, "mode": self.mode, "clip": None, "status": "applied"}

    # ---- playback (video track) ----
    def _schedule_auto(self) -> None:
        clips, lo, hi = MODES.get(self.mode, ((), 0, 0))
        if lo is None:
            lo, hi = self.listen_interval
        clips = [c for c in clips if self.available(c)]
        if not clips or hi <= 0:
            self.next_auto = None
            return
        self.next_auto = self.tick + int(self.rng.uniform(lo, hi) * FPS)
        self._auto_clips = clips

    def _idle_frame(self):
        n = len(self.a.frames)
        f = self.a.frames[ping_pong(self.pos, n)]
        self.pos += 1
        return f

    def frame(self, tick: int):
        if not self.a or not self.a.frames:
            return None
        self.tick = tick
        if self.last_tick is not None and tick != self.last_tick + 1:
            self.clip = None          # an utterance played in between: drop a half-played clip
            self.next_auto = None
        self.last_tick = tick
        if self.clip is not None:
            return self._clip_frame()
        busy = self.busy()
        if self.next_auto is None:
            self._schedule_auto()
        if self.next_auto is not None and tick >= self.next_auto:
            if self.pending is None and not busy:
                self.pending = {"clip": self.rng.choice(self._auto_clips), "state": None, "at": tick, "wait_from": None}
            self._schedule_auto()
        if self.pending is not None and not busy:
            p = self.pending
            if p["wait_from"] is None:
                p["wait_from"] = tick
                if (tick - p["at"]) / FPS > self.queue_s:      # waited behind speech for too long: stale
                    self.pending = None
                    if p["state"]:
                        self.on_event({"type": "state_applied", "state": p["state"], "clip": None, "skipped": "stale"})
                    return self._idle_frame()
            k = ping_pong(self.pos, len(self.a.frames))
            costs = self.a.cost[:, k]
            i = int(np.argmin(costs))
            if costs[i] <= self.THR0 + self.THR_SLOPE * (tick - p["wait_from"]) / FPS:
                self.pending = None
                self.clip = {"name": p["clip"], "anchor": i, "j": 0, "frames": self.a.clips[p["clip"]][i]}
                if p["state"]:
                    self.on_event({"type": "state_applied", "state": p["state"], "clip": p["clip"], "start_ms": tick * 40})
                return self._clip_frame()
        return self._idle_frame()

    def _clip_frame(self):
        c = self.clip
        frames = c["frames"]
        f = frames[c["j"]]
        if c["j"] < self.XFADE:                              # blend from the idle loop (which keeps running)
            w = (c["j"] + 1) / (self.XFADE + 1)
            f = cv2.addWeighted(f, w, self._idle_frame(), 1 - w, 0)
        c["j"] += 1
        if c["j"] >= len(frames):                            # resume the idle loop exactly where the clip ends
            self.pos = self.a.anchors[c["anchor"]] + len(frames)
            self.clip = None
        return f

    def status(self) -> dict:
        return {"mode": self.mode, "playing": self.clip["name"] if self.clip else None,
                "pending": self.pending["clip"] if self.pending else None,
                "clips": sorted(self.a.clips) if self.a else []}
