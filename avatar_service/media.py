"""aiortc tracks fed from a session Timeline. Both tracks share one start instant (the media clock)."""
from __future__ import annotations

import asyncio
import fractions
import time
from typing import Callable, Optional

import av
import numpy as np
from aiortc import MediaStreamTrack

from .audio import FPS, OUT_SR
from .session import Timeline

VIDEO_CLOCK = 90000
AUDIO_PTIME = 960  # 20 ms @ 48 kHz


class MediaClock:
    def __init__(self):
        self.t0: Optional[float] = None

    def start(self) -> float:
        if self.t0 is None:
            self.t0 = time.time()
        return self.t0


class AvatarVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, clock: MediaClock, timeline: Timeline, idle: Callable[[int], Optional[np.ndarray]],
                 placeholder: np.ndarray):
        super().__init__()
        self.clock, self.timeline, self.idle, self.placeholder = clock, timeline, idle, placeholder
        self.n = 0
        self.frames_sent = 0

    async def recv(self):
        t0 = self.clock.start()
        delay = t0 + self.n / FPS - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        img = self.timeline.next_video(self.idle)
        if img is None:
            img = self.placeholder
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = int(self.n * VIDEO_CLOCK / FPS)
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
        self.n += 1
        self.frames_sent += 1
        return frame


class AvatarAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, clock: MediaClock, timeline: Timeline):
        super().__init__()
        self.clock, self.timeline = clock, timeline
        self.n = 0

    async def recv(self):
        t0 = self.clock.start()
        delay = t0 + self.n * AUDIO_PTIME / OUT_SR - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        pcm = self.timeline.next_audio(AUDIO_PTIME)
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = OUT_SR
        frame.pts = self.n * AUDIO_PTIME
        frame.time_base = fractions.Fraction(1, OUT_SR)
        self.n += 1
        return frame
