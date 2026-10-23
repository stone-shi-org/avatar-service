"""Per-session media timeline (pure Python: no aiortc / GPU, so it is unit-testable).

Time is counted in video ticks (1 tick = 40 ms at 25 fps); audio sample s plays at tick s / 1920 (48 kHz).
Both WebRTC tracks are paced from the same start instant, so "tick" is one shared clock.

An utterance's audio is *released* (queued for the audio track) only once `preroll` frames are rendered,
and its start tick is fixed at that moment. Afterwards the video track always shows frame
`tick - start_tick`: late frames are dropped and missing frames hold the previous one, so a rendering
stall can never make the lips drift behind the voice. Utterances queued back-to-back start exactly
where the previous one ends.
"""
from __future__ import annotations

import collections
import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .audio import FPS, SAMPLES_PER_FRAME

_ids = itertools.count(1)


@dataclass
class Utterance:
    text: str
    audio48: np.ndarray            # int16 mono 48 kHz
    id: int = field(default_factory=lambda: next(_ids))
    lead_silence_ms: int = 0
    tts_ms: int = 0
    frames: "collections.deque" = field(default_factory=collections.deque)   # (idx, rgb) from the renderer thread
    rendered: int = 0
    start_tick: Optional[int] = None
    end_tick: Optional[int] = None    # known once rendering is done
    cancelled: bool = False
    t_created: float = field(default_factory=time.time)
    t_first_frame: Optional[float] = None
    t_released: Optional[float] = None
    t_done: Optional[float] = None

    @property
    def audio_frames(self) -> int:
        return math.ceil(len(self.audio48) / SAMPLES_PER_FRAME)

    def stats(self) -> dict:
        ms = lambda t: None if t is None else round((t - self.t_created) * 1000)  # noqa: E731
        return {"id": self.id, "text": self.text[:80], "tts_ms": self.tts_ms, "lead_silence_ms": self.lead_silence_ms,
                "audio_ms": round(len(self.audio48) / 48), "frames": self.rendered,
                "first_frame_ms": ms(self.t_first_frame), "released_ms": ms(self.t_released), "rendered_ms": ms(self.t_done),
                "start_tick": self.start_tick, "end_tick": self.end_tick, "cancelled": self.cancelled}


class Timeline:
    """Owned by the asyncio loop thread; the renderer thread only appends to `Utterance.frames`."""

    def __init__(self, preroll: int = 8, on_event: Callable[[dict], None] = lambda e: None):
        self.preroll = preroll
        self.on_event = on_event
        self.tick = 0                    # next video tick to be shown
        self.audio_q: "collections.deque[np.ndarray]" = collections.deque()
        self.audio_carry = np.zeros(0, np.int16)
        self.audio_emitted = 0           # samples handed to the audio track (incl. silence)
        self.audio_queued_end = 0        # sample position where queued audio ends
        self.utts: list[Utterance] = []
        self.last_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.history: "collections.deque[dict]" = collections.deque(maxlen=50)

    # ---- state ----
    @property
    def busy(self) -> bool:
        return bool(self.utts)

    def add(self, utt: Utterance) -> None:
        self.utts.append(utt)

    # ---- renderer side (called on the loop thread via call_soon_threadsafe) ----
    def frame_ready(self, utt: Utterance) -> None:
        if utt.t_first_frame is None:
            utt.t_first_frame = time.time()
        if utt.start_tick is None and not utt.cancelled and utt.rendered >= self.preroll:
            self.release(utt)

    def release(self, utt: Utterance) -> None:
        if utt.start_tick is not None or utt.cancelled:
            return
        with self.lock:
            cur = max(self.audio_emitted, self.audio_queued_end)
            start = max(self.tick + 1, math.ceil(cur / SAMPLES_PER_FRAME))
            for u in self.utts:                  # never overlap a previous utterance
                if u is utt:
                    break
                if u.end_tick is not None:
                    start = max(start, u.end_tick)
                elif u.start_tick is not None:   # previous still rendering: start after its audio
                    start = max(start, u.start_tick + u.audio_frames)
            gap = start * SAMPLES_PER_FRAME - cur
            if gap > 0:
                self.audio_q.append(np.zeros(gap, np.int16))
            self.audio_q.append(utt.audio48)
            self.audio_queued_end = start * SAMPLES_PER_FRAME + len(utt.audio48)
            utt.start_tick = start
            utt.t_released = time.time()
        self.on_event({"type": "utterance_start", "id": utt.id, "text": utt.text, "start_ms": start * 40,
                       "lead_silence_ms": utt.lead_silence_ms, "audio_ms": round(len(utt.audio48) / 48)})

    def finish(self, utt: Utterance, frames: int) -> None:
        """Rendering is done: make audio and video exactly the same length."""
        utt.t_done = time.time()
        if utt.cancelled:
            return
        if utt.start_tick is None:
            if frames == 0:
                self._drop(utt)
                return
            self.release(utt)
        total = max(frames, utt.audio_frames)
        with self.lock:
            pad = total * SAMPLES_PER_FRAME - len(utt.audio48)
            if pad > 0:
                self.audio_q.append(np.zeros(pad, np.int16))
                self.audio_queued_end += pad
        utt.end_tick = utt.start_tick + total

    def stop(self) -> list[int]:
        """Barge-in: drop everything queued and pending; returns the cancelled utterance ids."""
        ids = []
        with self.lock:
            for u in self.utts:
                u.cancelled = True
                ids.append(u.id)
                self.history.append(u.stats())
            self.utts.clear()
            self.audio_q.clear()
            self.audio_carry = np.zeros(0, np.int16)
            self.audio_queued_end = self.audio_emitted
        for i in ids:
            self.on_event({"type": "utterance_end", "id": i, "cancelled": True})
        return ids

    def _drop(self, utt: Utterance) -> None:
        if utt in self.utts:
            self.utts.remove(utt)
            self.history.append(utt.stats())
            self.on_event({"type": "utterance_end", "id": utt.id, "cancelled": utt.cancelled})

    # ---- track side ----
    def next_video(self, idle: Callable[[int], Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Frame for the current tick, then advance the tick."""
        t = self.tick
        self.tick += 1
        while self.utts and self.utts[0].end_tick is not None and t >= self.utts[0].end_tick:
            self._drop(self.utts[0])
        u = self.utts[0] if self.utts else None
        if u is None or u.start_tick is None or t < u.start_tick:
            f = idle(t)
            if f is not None:
                self.last_frame = f
            return self.last_frame
        want = t - u.start_tick
        frame = None
        while u.frames and u.frames[0][0] < want:
            u.frames.popleft()               # late: drop to stay in sync with the voice
        if u.frames and u.frames[0][0] == want:
            frame = u.frames.popleft()[1]
        if frame is not None:
            self.last_frame = frame
        return self.last_frame               # missing: hold the previous frame

    def next_audio(self, n: int) -> np.ndarray:
        with self.lock:
            buf = self.audio_carry
            while len(buf) < n and self.audio_q:
                buf = np.concatenate([buf, self.audio_q.popleft()])
            if len(buf) >= n:
                out, self.audio_carry = buf[:n], buf[n:]
            else:
                out, self.audio_carry = np.concatenate([buf, np.zeros(n - len(buf), np.int16)]), np.zeros(0, np.int16)
            self.audio_emitted += n
        return out


def ticks_to_seconds(ticks: int) -> float:
    return ticks / FPS
