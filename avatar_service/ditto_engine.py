"""Ditto (antgroup/ditto-talkinghead) wrapper: audio -> composited RGB frames at 25 fps.

One instance per process. Ditto's StreamSDK is a single pipeline (not re-entrant), so `render` must be
called from one thread at a time; `Renderer` serialises with a lock.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import io
import logging
import math
import os
import sys
import threading
import time
from typing import Callable

import numpy as np

from .audio import DITTO_SR, FPS
from .avatars import Avatar
from .gestures import IdleAssets, pick_anchors, pose_cost

log = logging.getLogger("avatar.ditto")

CHUNK = (3, 5, 2)  # HuBERT streaming chunk: 3 past, 5 current, 2 future frames (Ditto's online default)


# Ditto's motion is LivePortrait's: 21 implicit 3D keypoints; `exp` (1, 63) is a per-keypoint expression offset.
# For a portrait, MotionStitch takes `exp` from the audio->motion model only for these lip keypoints (and the eye
# ones); everything else stays at the source image (core/atomic_components/motion_stitch.py, `_lip` / `_eye`).
LIP_KP = (6, 12, 14, 17, 19, 20)
LIP_DIMS = np.array([3 * k + c for k in LIP_KP for c in range(3)])


def patch_motion_stitch(ms) -> None:
    """Give Ditto's MotionStitch instance extra per-frame controls (passed like Ditto's own `ctrl_info` kwargs).

    - `mouth_alpha`: scales the lip keypoints' motion relative to the stream's first frame `d0`. Ditto drives
      a portrait *relative* to d0 (x_s + (x_d - d0)), so 0 holds the lips exactly at the source image's
      (closed) mouth and 1 leaves the model's prediction untouched. Default: `ms.mouth_alpha` (1.0).
    - `add_pitch` / `add_yaw` / `add_roll` (degrees): added to the cfg's own `delta_*` (the online cfg sets
      `overall_ctrl_info = {"delta_pitch": 2}`; passing `delta_pitch` per frame would *replace* that).
    - `add_exp` (1, 63): added to the final expression (Ditto's `delta_exp`).
    - `eye_idx`: position in Ditto's scripted blink plan for this frame (re-rendering recorded frames).
    If `ms.recorder` is a list, every incoming (raw) motion frame is appended to it with its blink index.
    """
    base = type(ms)
    if getattr(base, "_avatar_patched", False):
        return

    class PatchedMotionStitch(base):
        _avatar_patched = True
        mouth_alpha = 1.0
        recorder = None

        def __call__(self, x_s_info, x_d_info, **kw):
            alpha = kw.pop("mouth_alpha", self.mouth_alpha)
            if "eye_idx" in kw:
                self.idx = kw.pop("eye_idx")
            overall = self.overall_ctrl_info or {}
            # plain floats / the motion's own dtype: a float64 here would turn the keypoints into float64, which
            # the TensorRT engines would read as garbage (and PyTorch rejects)
            for k in ("pitch", "yaw", "roll"):
                if f"add_{k}" in kw:
                    kw[f"delta_{k}"] = float(kw.get(f"delta_{k}", overall.get(f"delta_{k}", 0)) + kw.pop(f"add_{k}"))
            if "add_exp" in kw:
                kw["delta_exp"] = np.asarray(kw.pop("add_exp"), dtype=x_d_info["exp"].dtype).reshape(x_d_info["exp"].shape)
            if self.recorder is not None:
                self.recorder.append((copy.deepcopy(x_d_info), self.idx))
            if alpha < 1 and self.d0 is not None:
                exp = x_d_info["exp"].copy()
                d0 = self.d0["exp"].reshape(exp.shape)
                exp[:, LIP_DIMS] = d0[:, LIP_DIMS] + (exp[:, LIP_DIMS] - d0[:, LIP_DIMS]) * alpha
                x_d_info = {**x_d_info, "exp": exp}
            return super().__call__(x_s_info, x_d_info, **kw)

    ms.__class__ = PatchedMotionStitch


def _degrees(bins: np.ndarray) -> float:
    """Ditto pose logits (66 bins of 3 degrees) -> degrees (same as motion_stitch.bin66_to_degree)."""
    x = np.asarray(bins, np.float64).reshape(-1)
    if x.size != 66:
        return float(x[0])
    e = np.exp(x - x.max())
    return float((e / e.sum() * np.arange(66)).sum() * 3 - 97.5)


def motion_poses(motion: list) -> np.ndarray:
    """(N, 5) pitch, yaw, roll (degrees), tx, ty of recorded raw motion frames."""
    return np.array([[_degrees(d["pitch"]), _degrees(d["yaw"]), _degrees(d["roll"]),
                      float(d["t"].reshape(-1)[0]), float(d["t"].reshape(-1)[1])] for d, _ in motion])


class _NullWriter:
    """Replaces Ditto's imageio MP4 writer: frames go to our callback instead."""

    def __init__(self, *a, **k):
        self.cb: Callable[[np.ndarray], None] | None = None

    def __call__(self, img, fmt="rgb"):
        if self.cb is not None:
            self.cb(img)

    def close(self):
        pass


class TurnFeed:
    """Sentences of one speaking turn, handed from the API to the render thread.

    `put` fails once the stream has closed, so the caller starts a new stream instead: a sentence can never
    be dropped by a stream that is just shutting down (`close_if_empty` is atomic with `put`).
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.items: list = []
        self.closed = False
        self.cancelled = False
        self.taken = 0      # items handed out by `next` (a consumer in another process checks it, see worker.py)

    def put(self, item) -> bool:
        with self.cond:
            if self.closed:
                return False
            self.items.append(item)
            self.cond.notify()
            return True

    def next(self, timeout: float):
        with self.cond:
            if not self.items and timeout > 0 and not self.cancelled:
                self.cond.wait(timeout)
            if not self.items or self.cancelled:
                return None
            self.taken += 1
            return self.items.pop(0)

    def close_if_empty(self) -> bool:
        with self.cond:
            if not self.items or self.cancelled:
                self.closed = True
            return self.closed

    def close_if_drained(self, received: int) -> bool:
        """`close_if_empty` for a remote consumer that has received `received` items: False while one is in flight."""
        with self.cond:
            if self.taken > received and not self.cancelled:
                return False
            return self.close_if_empty()

    def cancel(self) -> list:
        with self.cond:
            self.cancelled = self.closed = True
            dropped, self.items = self.items, []
            self.cond.notify_all()
            return dropped


SAMPLES_PER_VIDEO_FRAME = DITTO_SR // FPS   # 640
FLUSH_FRAMES = 12   # silence appended when the queue runs dry: pushes the last speech frames out of the pipeline
FLUSH_WAIT_S = 0.15


class DittoRenderer:
    def __init__(self, ditto_dir: str, ditto_models: str, data_root: str, sampling_steps: int = 10):
        if ditto_dir not in sys.path:
            sys.path.insert(0, ditto_dir)
        import stream_pipeline_online as spo  # noqa: E402  (Ditto repo)

        spo.VideoWriterByImageIO = _NullWriter  # never spawn ffmpeg
        cfg = os.path.join(ditto_models, "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            self.sdk = spo.StreamSDK(cfg, data_root)
        log.info("Ditto loaded in %.1f s (engines: %s)", time.time() - t, data_root)
        self.steps = sampling_steps
        self.lock = threading.Lock()
        self._registered: dict[str, object] = {}
        reg = self.sdk.avatar_registrar

        def cached(src, **kw):  # avatar registration (face detect + appearance) once per image
            if src not in self._registered:
                self._registered[src] = reg(src, **kw)
            return self._registered[src]

        self.sdk.avatar_registrar = cached
        patch_motion_stitch(self.sdk.motion_stitch)

    def render(self, avatar: Avatar, audio16k: np.ndarray, on_frame: Callable[[np.ndarray], None],
               cancelled: Callable[[], bool] = lambda: False, emo: int | None = None) -> int:
        """Render `audio16k` (float32 mono 16 kHz) as fast as the GPU allows; returns the frame count."""
        with self.lock:
            return self._render(avatar, audio16k, on_frame, cancelled, emo)

    def _render(self, avatar, audio16k, on_frame, cancelled=lambda: False, emo=None) -> int:
        sdk = self.sdk
        count = [0]

        def emit(img):
            if cancelled():
                return
            count[0] += 1
            on_frame(np.ascontiguousarray(img))

        kw = dict(online_mode=True, sampling_timesteps=self.steps, emo=avatar.emo if emo is None else emo)
        with contextlib.redirect_stdout(io.StringIO()):
            sdk.setup(avatar.image, "/tmp/avatar-unused.mp4", **kw)
        sdk.writer_pbar.disable = True
        sdk.writer.cb = emit
        sdk.setup_Nd(N_d=math.ceil(len(audio16k) / DITTO_SR * FPS))
        a = np.concatenate([np.zeros(CHUNK[0] * 640, np.float32), audio16k.astype(np.float32)])
        split = int(sum(CHUNK) * 0.04 * DITTO_SR) + 80
        try:
            for i in range(0, len(a), CHUNK[1] * 640):
                if cancelled():
                    break
                ch = a[i:i + split]
                if len(ch) < split:
                    ch = np.pad(ch, (0, split - len(ch)))
                sdk.run_chunk(ch, CHUNK)
        finally:
            sdk.close()
        return count[0]

    def warm(self, avatar: Avatar) -> None:
        """One short render: the first one after loading also sets up the TensorRT contexts."""
        self.render(avatar, np.zeros(DITTO_SR, np.float32), lambda f: None)

    def stream(self, avatar: Avatar, feed: TurnFeed, on_segment: Callable[[object, int, int], None],
               on_frame: Callable[[int, np.ndarray], None], emo: int | None = None, linger_s: float = 1.0) -> None:
        """Render a whole speaking turn as ONE continuous Ditto stream.

        Sentences are taken from `feed` as they arrive and appended to the running stream's audio, so only the
        first sentence pays Ditto's start-up (setup + two diffusion windows, ~300-450 ms); later ones continue
        seamlessly. Output frame g corresponds to audio samples [g*640, (g+1)*640) of the stream.
        `on_segment(key, first_frame, n_frames)` is called before a sentence's frames can appear.
        The stream ends when nothing new arrives within `linger_s` after the queue ran dry.
        """
        with self.lock:
            sdk = self.sdk
            g = [0]

            def emit(img):
                if not feed.cancelled:
                    on_frame(g[0], np.ascontiguousarray(img))
                g[0] += 1

            kw = dict(online_mode=True, sampling_timesteps=self.steps, emo=avatar.emo if emo is None else emo)
            with contextlib.redirect_stdout(io.StringIO()):
                sdk.setup(avatar.image, "/tmp/avatar-unused.mp4", **kw)
            sdk.writer_pbar.disable = True
            sdk.writer.cb = emit
            sdk.setup_Nd(N_d=-1)   # open-ended stream (Ditto then cycles a 3000-frame blink plan; a huge N_d costs ~0.5 s per turn)
            hop = CHUNK[1] * SAMPLES_PER_VIDEO_FRAME
            split = int(sum(CHUNK) * 0.04 * DITTO_SR) + 80
            buf = np.zeros(CHUNK[0] * SAMPLES_PER_VIDEO_FRAME, np.float32)   # past-context prefix
            pos, frames_in, flushed = 0, 0, True

            def run_available():
                nonlocal buf, pos
                while len(buf) - pos >= split and not feed.cancelled:
                    sdk.run_chunk(buf[pos:pos + split], CHUNK)
                    pos += hop
                if pos > 20 * hop:
                    buf, pos = buf[pos:], 0

            try:
                while not feed.cancelled:
                    # the next sentence's TTS usually lands within ~200 ms: wait a little before flushing
                    item = feed.next(0 if flushed else FLUSH_WAIT_S)
                    if item is None:
                        if not flushed:
                            buf = np.concatenate([buf, np.zeros(FLUSH_FRAMES * SAMPLES_PER_VIDEO_FRAME, np.float32)])
                            frames_in += FLUSH_FRAMES
                            flushed = True
                            run_available()
                            continue
                        item = feed.next(linger_s)
                        if item is None:
                            if feed.close_if_empty():
                                break
                            continue
                    key, a16 = item
                    n = math.ceil(len(a16) / SAMPLES_PER_VIDEO_FRAME)
                    a16 = np.pad(a16.astype(np.float32), (0, n * SAMPLES_PER_VIDEO_FRAME - len(a16)))
                    on_segment(key, frames_in, n)
                    buf = np.concatenate([buf, a16])
                    frames_in += n
                    flushed = False
                    run_available()
            finally:
                sdk.close()

    def render_idle(self, avatar: Avatar, seconds: int, still_mouth: bool = True,
                    clips: dict[str, list[dict]] | None = None, anchors: int = 2) -> IdleAssets:
        """Idle loop: the portrait animated over near-silence (blinks, slight head motion), plus gesture clips.

        Ditto's audio->motion model still predicts small lip movements from silence, so with `still_mouth` the
        lips are held at the portrait's own closed mouth (`mouth_alpha=0`); blinks and head sway are untouched.

        `clips` ({name: per-frame ctrl dicts}) are re-rendered from the idle loop's own recorded motion, starting
        at each of `anchors` splice frames, with the gesture's controls on top. Controls are zero at both ends, so
        a clip starts on exactly its anchor frame and ends on exactly the idle frame it was cut up to.
        """
        rng = np.random.default_rng(1)
        noise = (rng.normal(0, 30, DITTO_SR * seconds) / 32768.0).astype(np.float32)
        frames: list[np.ndarray] = []
        with self.lock:
            ms = self.sdk.motion_stitch
            ms.mouth_alpha, ms.recorder = (0.0 if still_mouth else 1.0), []
            try:
                self._render(avatar, noise, frames.append)
                motion = ms.recorder
            finally:
                ms.mouth_alpha, ms.recorder = 1.0, None
            out = IdleAssets(frames)
            if not clips:
                return out
            if len(motion) != len(frames):
                log.warning("idle motion not recorded (%d vs %d frames): no gesture clips", len(motion), len(frames))
                return out
            eye_plan = getattr(ms, "delta_eye_idx_list", None) if ms.drive_eye and ms.delta_eye_arr is not None else None
            blink = np.array([bool(eye_plan[i % len(eye_plan)]) if eye_plan else False for _, i in motion])
            poses = motion_poses(motion)
            out.anchors = pick_anchors(poses, blink, anchors, max(len(c) for c in clips.values()))
            out.cost = pose_cost(poses, blink, out.anchors).astype(np.float32)
            base = {"mouth_alpha": 0.0 if still_mouth else 1.0}
            for name, ctrls in clips.items():
                out.clips[name] = [self.render_motion(motion[a:a + len(ctrls)], [{**base, **c} for c in ctrls])
                                   for a in out.anchors]
            return out

    def render_motion(self, motion: list, ctrls: list[dict]) -> list[np.ndarray]:
        """Re-render recorded motion frames synchronously (stitch -> warp -> decode -> paste back), with per-frame
        controls. Must run right after a render of the same avatar (uses the SDK's source info and d0)."""
        sdk = self.sdk
        src = sdk.source_info
        x_s_info, f_s, img, m_c2o = src["x_s_info_lst"][0], src["f_s_lst"][0], src["img_rgb_lst"][0], src["M_c2o_lst"][0]
        # GPU stages here, the CPU paste-back (cv2 warp + blend at full size) overlapped in a second thread;
        # decode_f3d returns a fresh array, so the next frame can't overwrite it
        with concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="putback") as pool:
            futs = []
            for (x_d_info, eye_idx), ctrl in zip(motion, ctrls):
                x_s, x_d = sdk.motion_stitch(x_s_info, copy.deepcopy(x_d_info), eye_idx=eye_idx, **ctrl)
                render = sdk.decode_f3d(sdk.warp_f3d(f_s, x_s, x_d))
                futs.append(pool.submit(sdk.putback, img, render, m_c2o))
            return [np.ascontiguousarray(f.result()) for f in futs]


class FakeRenderer:
    """GPU-free stand-in for tests: the portrait with a bar whose height follows the audio level."""

    def __init__(self, *a, **k):
        self.lock = threading.Lock()

    def render(self, avatar, audio16k, on_frame, cancelled=lambda: False, emo=None) -> int:
        with self.lock:
            return self._frames(avatar, audio16k, on_frame, cancelled)

    def _frames(self, avatar, audio16k, on_frame, cancelled) -> int:
        import cv2
        img = cv2.cvtColor(cv2.imread(avatar.image), cv2.COLOR_BGR2RGB) if os.path.isfile(avatar.image) \
            else np.full((360, 640, 3), 90, np.uint8)
        n = math.ceil(len(audio16k) / DITTO_SR * FPS)
        hop = DITTO_SR // FPS
        for i in range(n):
            if cancelled():
                break
            lvl = float(np.sqrt(np.mean(audio16k[i * hop:(i + 1) * hop] ** 2) + 1e-12))
            f = img.copy()
            h = min(f.shape[0] - 1, int(lvl * 4 * f.shape[0]))
            f[f.shape[0] - h:, :16] = (255, 255, 255)
            on_frame(f)
            time.sleep(0.004)
        return n

    def warm(self, avatar) -> None:
        self.render(avatar, np.zeros(DITTO_SR, np.float32), lambda f: None)

    def stream(self, avatar, feed, on_segment, on_frame, emo=None, linger_s: float = 1.0) -> None:
        with self.lock:
            g = 0
            while not feed.cancelled:
                item = feed.next(linger_s)
                if item is None:
                    if feed.close_if_empty():
                        break
                    continue
                key, a16 = item
                n = math.ceil(len(a16) / (DITTO_SR // FPS))
                on_segment(key, g, n)
                k = [g]

                def emit(f):
                    if not feed.cancelled:
                        on_frame(k[0], f)
                    k[0] += 1

                self._frames(avatar, a16, emit, lambda: feed.cancelled)
                g += n

    def render_idle(self, avatar, seconds, still_mouth=True, clips=None, anchors=2):
        frames: list[np.ndarray] = []
        self.render(avatar, np.zeros(DITTO_SR, np.float32), frames.append)
        out = IdleAssets(frames)
        if clips:
            n = len(frames)
            out.anchors = [0]
            out.cost = np.zeros((1, n), np.float32)
            for name, ctrls in clips.items():   # the portrait with a coloured bar on the right, per clip
                f = frames[0].copy()
                f[:, -16:] = (40 * (1 + list(clips).index(name)) % 256, 200, 80)
                out.clips[name] = [[f] * min(len(ctrls), n - 1)]
        return out
