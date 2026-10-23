"""Service core: renderer lifecycle, idle loops, sessions, speaker admission."""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import pickle
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .audio import DITTO_SR, OUT_SR, lead_silence_ms, resample, to_int16
from .avatars import Avatar, load_catalog
from .config import Settings
from .session import Timeline, Utterance

log = logging.getLogger("avatar.service")


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def load_portrait(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ServiceError(500, "bad_avatar", f"cannot read {os.path.basename(path)}")
    h, w = img.shape[:2]
    return np.ascontiguousarray(cv2.cvtColor(img[: h - h % 2, : w - w % 2], cv2.COLOR_BGR2RGB))  # even size for yuv420


class IdleLoops:
    """Pre-rendered idle animation per avatar (rendered once, cached on the data volume as JPEGs)."""

    def __init__(self, data_dir: str, seconds: int):
        self.dir = os.path.join(data_dir, "idle")
        self.seconds = seconds
        self.frames: dict[str, list[np.ndarray]] = {}
        self.pending: set[str] = set()
        self.lock = threading.Lock()

    def _path(self, av: Avatar) -> str:
        h = hashlib.sha1(open(av.image, "rb").read() + f"{self.seconds}:{av.emo}:v1".encode()).hexdigest()[:12]
        return os.path.join(self.dir, f"{av.id}-{h}.pkl")

    def load_cached(self, av: Avatar) -> bool:
        p = self._path(av)
        if not os.path.isfile(p):
            return False
        with open(p, "rb") as f:
            jpgs = pickle.load(f)   # written by us (below), on our own volume
        self.frames[av.id] = [cv2.cvtColor(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for j in jpgs]
        return True

    def store(self, av: Avatar, frames: list[np.ndarray]) -> None:
        frames = [f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2] for f in frames]
        self.frames[av.id] = frames
        os.makedirs(self.dir, exist_ok=True)
        jpgs = [cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for f in frames]
        tmp = self._path(av) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(jpgs, f)
        os.replace(tmp, self._path(av))

    def frame(self, avatar_id: str, tick: int) -> Optional[np.ndarray]:
        fr = self.frames.get(avatar_id)
        if not fr:
            return None
        n = len(fr)
        if n == 1:
            return fr[0]
        k = tick % (2 * n - 2)           # ping-pong: seamless without a cross-fade
        return fr[k if k < n else 2 * n - 2 - k]


@dataclass
class Session:
    id: str
    avatar: Avatar
    voice: str
    portrait: np.ndarray
    timeline: Timeline
    pc: object = None
    channel: object = None
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    connection_state: str = "new"
    video_track: object = None
    feed: object = None                  # TurnFeed of the turn being rendered
    turn_lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict:
        return {"session_id": self.id, "avatar": self.avatar.id, "voice": self.voice,
                "connection_state": self.connection_state, "speaking": self.timeline.busy,
                "frame_size": [int(self.portrait.shape[1]), int(self.portrait.shape[0])],
                "frames_sent": getattr(self.video_track, "frames_sent", 0),
                "utterances": [u.stats() for u in self.timeline.utts] + list(self.timeline.history)[-10:]}


class AvatarService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.state = "starting"
        self.error: Optional[str] = None
        self.engines_dir: Optional[str] = None
        self.gpu: Optional[str] = None
        self.renderer = None
        self.catalog: dict[str, Avatar] = load_catalog(settings.avatars_dir)
        self.idle = IdleLoops(settings.data_dir, settings.idle_loop_seconds)
        self.sessions: dict[str, Session] = {}
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    # ---------- startup ----------
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        threading.Thread(target=self._init, name="avatar-init", daemon=True).start()

    def _init(self) -> None:
        try:
            if self.s.fake_renderer:
                from .ditto_engine import FakeRenderer
                self.renderer, self.gpu = FakeRenderer(), "fake"
            else:
                from .engines import ensure_engines
                from .ditto_engine import DittoRenderer
                self.state = "building_engines"
                self.engines_dir = ensure_engines(self.s.ditto_models, self.s.data_dir,
                                                  native=os.environ.get("AVATAR_NATIVE_ENGINES", "1") != "0")
                self.state = "loading"
                import torch
                self.gpu = torch.cuda.get_device_name(0)
                self.renderer = DittoRenderer(self.s.ditto_dir, self.s.ditto_models, self.engines_dir, self.s.sampling_steps)
            self.state = "warming_up"
            for av in self.catalog.values():
                if not self.idle.load_cached(av):
                    t = time.time()
                    self.idle.store(av, self.renderer.render_idle(av, self.s.idle_loop_seconds))
                    log.info("idle loop for %s rendered in %.1f s", av.id, time.time() - t)
            if self.catalog:  # first real render also warms the TensorRT contexts
                av = next(iter(self.catalog.values()))
                self.renderer.render(av, np.zeros(DITTO_SR, np.float32), lambda f: None)
            self.state = "ready"
            log.info("ready: %d avatar(s) %s, gpu=%s", len(self.catalog), list(self.catalog), self.gpu)
        except Exception as e:  # noqa: BLE001 - surfaced on /health
            log.exception("startup failed")
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    def health(self) -> dict:
        return {"status": self.state, "error": self.error, "gpu": self.gpu, "engines": self.engines_dir,
                "avatars": list(self.catalog), "sessions": len(self.sessions),
                "speaking": sum(1 for x in self.sessions.values() if x.timeline.busy),
                "max_sessions": self.s.max_sessions, "max_speakers": self.s.max_speakers}

    def require_ready(self) -> None:
        if self.state != "ready":
            raise ServiceError(503, "not_ready", f"service is {self.state}" + (f": {self.error}" if self.error else ""))

    # ---------- sessions ----------
    def create_session(self, avatar_id: Optional[str], voice: Optional[str]) -> Session:
        self.require_ready()
        self.expire()
        if len(self.sessions) >= self.s.max_sessions:
            raise ServiceError(429, "too_many_sessions", f"at most {self.s.max_sessions} sessions")
        if not self.catalog:
            raise ServiceError(500, "no_avatars", "no avatars on the model volume")
        aid = avatar_id or next(iter(self.catalog))
        av = self.catalog.get(aid)
        if av is None:
            raise ServiceError(404, "unknown_avatar", f"unknown avatar {aid!r}; have {list(self.catalog)}")
        sid = "ses_" + secrets.token_hex(8)
        sess = Session(sid, av, voice or av.voice, load_portrait(av.image),
                       Timeline(preroll=self.s.preroll_frames, trim_lead=self.s.trim_lead_silence))
        sess.timeline.on_event = lambda e, s=sess: self._send_event(s, e)
        self.sessions[sid] = sess
        return sess

    def get(self, sid: str) -> Session:
        sess = self.sessions.get(sid)
        if sess is None:
            raise ServiceError(404, "unknown_session", "unknown session")
        sess.last_used = time.time()
        return sess

    async def close_session(self, sid: str) -> None:
        sess = self.sessions.pop(sid, None)
        if sess is None:
            raise ServiceError(404, "unknown_session", "unknown session")
        self.stop(sess)
        if sess.pc is not None:
            await sess.pc.close()

    def expire(self) -> None:
        now = time.time()
        for sid, sess in list(self.sessions.items()):
            dead = sess.connection_state in ("failed", "closed")
            if dead or now - sess.last_used > self.s.session_idle_timeout_s:
                self.sessions.pop(sid, None)
                self.stop(sess)
                if sess.pc is not None and self.loop:
                    self.loop.create_task(sess.pc.close())

    def _send_event(self, sess: Session, event: dict) -> None:
        ch = sess.channel
        if ch is not None and getattr(ch, "readyState", "") == "open":
            import json
            ch.send(json.dumps(event))

    # ---------- speaking ----------
    def admit(self, sess: Session) -> None:
        if sess.timeline.busy:
            return
        speaking = sum(1 for x in self.sessions.values() if x.timeline.busy and x is not sess)
        if speaking >= self.s.max_speakers:
            raise ServiceError(409, "busy", "another session is speaking; try again or fall back")

    def speak(self, sess: Session, audio: np.ndarray, sr: int, text: str, tts_ms: int, emo: Optional[int]) -> Utterance:
        """Queue `audio` (float32 mono, any rate) on the session's current speaking turn. Returns at once.

        All sentences of a turn go through one continuous Ditto stream (see DittoRenderer.stream), so only the
        first pays the renderer's start-up and the rest follow without gaps.
        """
        from .ditto_engine import TurnFeed

        if len(audio) / sr > self.s.max_audio_seconds:
            raise ServiceError(413, "audio_too_long", f"at most {self.s.max_audio_seconds} s per utterance")
        a48f = resample(audio, sr, OUT_SR)
        utt = Utterance(text=text, audio48=to_int16(a48f), lead_silence_ms=lead_silence_ms(a48f, OUT_SR), tts_ms=tts_ms)
        a16 = resample(audio, sr, DITTO_SR)
        sess.timeline.add(utt)
        with sess.turn_lock:
            if sess.feed is not None and sess.feed.put((utt, a16)):
                return utt                       # appended to the running stream
            feed = sess.feed = TurnFeed()
            feed.put((utt, a16))
        self.pool.submit(self._run_turn, sess, feed, emo)
        return utt

    def _run_turn(self, sess: Session, feed, emo: Optional[int]) -> None:
        tl, loop = sess.timeline, self.loop
        segs: list[list] = []                    # [first_frame, n_frames, utt, done]

        def on_segment(utt, first, n):
            segs.append([first, n, utt, False])

        def on_frame(g, img):
            for seg in segs:
                first, n, utt, done = seg
                if first <= g < first + n:
                    if utt.cancelled:
                        return
                    h, w = img.shape[:2]
                    utt.frames.append((g - first, img[: h - h % 2, : w - w % 2]))
                    utt.rendered += 1
                    loop.call_soon_threadsafe(tl.frame_ready, utt)
                    if g == first + n - 1:
                        seg[3] = True
                        loop.call_soon_threadsafe(tl.finish, utt, n)
                    return
            # frames of the flush silence between sentences belong to nobody

        try:
            if feed.cancelled:
                return
            self.renderer.stream(sess.avatar, feed, on_segment, on_frame, emo=emo)
        except Exception:  # noqa: BLE001
            log.exception("render stream failed for session %s", sess.id)
        finally:
            with sess.turn_lock:
                if sess.feed is feed:
                    sess.feed = None
                for utt, _ in feed.cancel():     # anything still queued (only after an error)
                    utt.cancelled = True
            for first, n, utt, done in segs:
                if not done:
                    loop.call_soon_threadsafe(tl.finish, utt, utt.rendered)

    def stop(self, sess: Session) -> list[int]:
        """Barge-in: cancel the running stream and everything queued or playing."""
        with sess.turn_lock:
            if sess.feed is not None:
                sess.feed.cancel()
                sess.feed = None
        return sess.timeline.stop()
