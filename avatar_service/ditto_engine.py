"""Ditto (antgroup/ditto-talkinghead) wrapper: audio -> composited RGB frames at 25 fps.

One instance per process. Ditto's StreamSDK is a single pipeline (not re-entrant), so `render` must be
called from one thread at a time; `Renderer` serialises with a lock.
"""
from __future__ import annotations

import contextlib
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

log = logging.getLogger("avatar.ditto")

CHUNK = (3, 5, 2)  # HuBERT streaming chunk: 3 past, 5 current, 2 future frames (Ditto's online default)


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
            return self.items.pop(0) if self.items and not self.cancelled else None

    def close_if_empty(self) -> bool:
        with self.cond:
            if not self.items or self.cancelled:
                self.closed = True
            return self.closed

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

    def render(self, avatar: Avatar, audio16k: np.ndarray, on_frame: Callable[[np.ndarray], None],
               cancelled: Callable[[], bool] = lambda: False, emo: int | None = None) -> int:
        """Render `audio16k` (float32 mono 16 kHz) as fast as the GPU allows; returns the frame count."""
        with self.lock:
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

    def render_idle(self, avatar: Avatar, seconds: int) -> list[np.ndarray]:
        """Idle loop: the portrait animated over near-silence (blinks, slight head motion)."""
        rng = np.random.default_rng(1)
        noise = (rng.normal(0, 30, DITTO_SR * seconds) / 32768.0).astype(np.float32)
        frames: list[np.ndarray] = []
        self.render(avatar, noise, frames.append)
        return frames


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

    def render_idle(self, avatar, seconds):
        frames: list[np.ndarray] = []
        self.render(avatar, np.zeros(DITTO_SR, np.float32), frames.append)
        return frames
