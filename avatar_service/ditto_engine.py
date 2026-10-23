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


class Cancelled(Exception):
    pass


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
        import cv2
        img = cv2.cvtColor(cv2.imread(avatar.image), cv2.COLOR_BGR2RGB) if os.path.isfile(avatar.image) \
            else np.full((360, 640, 3), 90, np.uint8)
        n = math.ceil(len(audio16k) / DITTO_SR * FPS)
        hop = DITTO_SR // FPS
        with self.lock:
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

    def render_idle(self, avatar, seconds):
        frames: list[np.ndarray] = []
        self.render(avatar, np.zeros(DITTO_SR, np.float32), frames.append)
        return frames
