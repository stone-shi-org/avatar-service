"""Service core: renderer lifecycle, idle loops, sessions, speaker admission."""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import pickle
import re
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
from .gestures import STATES, Director, IdleAssets, clip_controls
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


def _jpg(f: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def _unjpg(j: bytes) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _even(f: np.ndarray) -> np.ndarray:
    return f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2]


class IdleLoops:
    """Pre-rendered idle animation + gesture clips per avatar (rendered once, cached on the data volume as JPEGs)."""

    VERSION = 2

    def __init__(self, data_dir: str, seconds: int, still_mouth: bool = True, clips: Optional[dict] = None,
                 anchors: int = 2):
        self.dir = os.path.join(data_dir, "idle")
        self.seconds = seconds
        self.still_mouth = still_mouth
        self.clips = clips or {}            # name -> per-frame controls (gestures.clip_controls)
        self.anchors = anchors
        self.assets: dict[str, IdleAssets] = {}
        self.lock = threading.Lock()

    def _path(self, av: Avatar) -> str:
        spec = repr((self.VERSION, self.seconds, av.emo, self.still_mouth, self.anchors,
                     sorted((k, repr(v)) for k, v in self.clips.items())))
        h = hashlib.sha1(open(av.image, "rb").read() + spec.encode()).hexdigest()[:12]
        return os.path.join(self.dir, f"{av.id}-{h}.pkl")

    def load_cached(self, av: Avatar) -> bool:
        p = self._path(av)
        if not os.path.isfile(p):
            return False
        with open(p, "rb") as f:
            d = pickle.load(f)   # written by us (below), on our own volume
        self.assets[av.id] = IdleAssets([_unjpg(j) for j in d["frames"]],
                                        {k: [[_unjpg(j) for j in v] for v in vs] for k, vs in d["clips"].items()},
                                        list(d["anchors"]), d["cost"])
        return True

    def store(self, av: Avatar, assets: IdleAssets) -> None:
        assets.frames = [_even(f) for f in assets.frames]
        assets.clips = {k: [[_even(f) for f in v] for v in vs] for k, vs in assets.clips.items()}
        self.assets[av.id] = assets
        os.makedirs(self.dir, exist_ok=True)
        d = {"frames": [_jpg(f) for f in assets.frames],
             "clips": {k: [[_jpg(f) for f in v] for v in vs] for k, vs in assets.clips.items()},
             "anchors": assets.anchors, "cost": assets.cost}
        path = self._path(av)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(d, f)
        os.replace(path + ".tmp", path)
        for fn in os.listdir(self.dir):     # older renders of this avatar (other settings / format)
            if re.fullmatch(re.escape(av.id) + r"-[0-9a-f]{12}\.pkl", fn) and os.path.join(self.dir, fn) != path:
                os.remove(os.path.join(self.dir, fn))


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
    director: Optional[Director] = None  # idle loop + gestures
    turn_lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict:
        return {"session_id": self.id, "avatar": self.avatar.id, "voice": self.voice,
                "connection_state": self.connection_state, "speaking": self.timeline.busy,
                "frame_size": [int(self.portrait.shape[1]), int(self.portrait.shape[0])],
                "frames_sent": getattr(self.video_track, "frames_sent", 0),
                "gesture": self.director.status() if self.director else None,
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
        clips = {c: clip_controls(c, settings.gesture_scale) for c in settings.gesture_clips} if settings.gestures else {}
        self.idle = IdleLoops(settings.data_dir, settings.idle_loop_seconds, settings.still_mouth_idle, clips,
                              settings.gesture_anchors)
        self.sessions: dict[str, Session] = {}
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.http = None                                # aiohttp.ClientSession, set by the server on startup
        self._warmed_avatars: set[str] = set()          # avatar ids rendered once since the renderer (re)loaded
        self._tts_warmed_at: dict[str, float] = {}      # voice -> time of the last successful TTS probe
        self._tts_warming: dict[str, asyncio.Future] = {}  # voice -> probe in flight

    # ---------- startup ----------
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        threading.Thread(target=self._init, name="avatar-init", daemon=True).start()

    def _worker_cfg(self) -> dict:
        return {"fake": self.s.fake_renderer, "ditto_dir": self.s.ditto_dir, "ditto_models": self.s.ditto_models,
                "data_dir": self.s.data_dir, "sampling_steps": self.s.sampling_steps,
                "native": os.environ.get("AVATAR_NATIVE_ENGINES", "1") != "0",
                "log_level": os.environ.get("AVATAR_LOG_LEVEL", "INFO")}

    @property
    def on_demand(self) -> bool:
        return self.s.model_idle_unload_s > 0

    def _on_model_state(self, state: str) -> None:
        if state == "unloaded":
            self._warmed_avatars.clear()   # a reloaded renderer has cold TensorRT contexts again
        if self.state in ("starting", "building_engines", "loading") and state in ("building_engines", "loading"):
            self.state = state      # startup progress on /health

    def _init(self) -> None:
        try:
            if self.on_demand:
                # renderer in a child process (worker.py), stopped after model_idle_unload_s without speech
                from .worker import OnDemandRenderer
                self.renderer = OnDemandRenderer(self._worker_cfg(), self.s.model_idle_unload_s, self._on_model_state)
                self.renderer.load()
                self.gpu, self.engines_dir = self.renderer.gpu, self.renderer.engines
            elif self.s.fake_renderer:
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
                    assets = self.renderer.render_idle(av, self.s.idle_loop_seconds, still_mouth=self.idle.still_mouth,
                                                       clips=self.idle.clips, anchors=self.idle.anchors)
                    self.idle.store(av, assets)
                    log.info("idle loop for %s rendered in %.1f s (%d frames; clips %s at anchors %s)", av.id,
                             time.time() - t, len(assets.frames),
                             {k: assets.clip_len(k) for k in assets.clips}, assets.anchors)
            if self.catalog:  # first real render also warms the TensorRT contexts for all avatars
                for av in self.catalog.values():
                    t_w = time.time()
                    self.renderer.warm(av)
                    self._warmed_avatars.add(av.id)
                    log.info("warmup render for avatar %s complete in %.2f s", av.id, time.time() - t_w)
            if self.s.tts_url:
                import urllib.request, json
                voices = {av.voice for av in self.catalog.values() if av.voice}
                for v in sorted(voices):
                    try:
                        req = urllib.request.Request(
                            f"{self.s.tts_url.rstrip('/')}/v1/audio/speech",
                            data=json.dumps({"model": self.s.tts_model, "voice": v, "input": "."}).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        urllib.request.urlopen(req, timeout=5)
                        self._tts_warmed_at[v] = time.time()
                        log.info("warmed TTS voice %s", v)
                    except Exception as e:
                        log.warning("could not warm TTS voice %s: %s", v, e)
            self.state = "ready"
            log.info("ready: %d avatar(s) %s, gpu=%s", len(self.catalog), list(self.catalog), self.gpu)
        except Exception as e:  # noqa: BLE001 - surfaced on /health
            log.exception("startup failed")
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    def health(self) -> dict:
        return {"status": self.state, "error": self.error, "gpu": self.gpu, "engines": self.engines_dir,
                "avatars": list(self.catalog), "sessions": len(self.sessions),
                "speaking": sum(1 for x in self.sessions.values() if x.timeline.busy),
                "max_sessions": self.s.max_sessions, "max_speakers": self.s.max_speakers,
                "model": self._model_status()}

    def _model_status(self) -> dict:
        r = self.renderer
        if not self.on_demand or r is None:
            return {"state": "loaded" if r is not None else "unloaded", "idle_unload_s": 0}
        return {"state": r.state, "idle_unload_s": self.s.model_idle_unload_s, "pid": r.proc.pid if r.proc else None,
                "idle_s": round(time.time() - r.last_used, 1), "loads": r.loads, "error": r.error}

    def preload(self, sess: Optional[Session] = None) -> None:
        """A client is about to need the renderer: start it now if it was unloaded (no-op when loaded)."""
        if self.on_demand and self.state == "ready":
            self.renderer.preload(sess.avatar if sess else None)

    def shutdown(self) -> None:
        if self.on_demand and self.renderer is not None:
            self.renderer.close()

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
        sess.director = Director(self.idle.assets.get(av.id), busy=lambda tl=sess.timeline: tl.busy,
                                 on_event=lambda e, s=sess: self._send_event(s, e),
                                 listen_interval=(self.s.listen_gesture_min_s, self.s.listen_gesture_max_s))
        self.sessions[sid] = sess
        self.preload(sess)
        if self._tts_needs_warm(sess.voice) and self.loop is not None:
            # fire and forget: wakes the TTS server (live-stt-2 evicts Kokoro after 10 min idle) while the
            # client is still negotiating WebRTC, so the first /say doesn't pay for the model load
            self.loop.create_task(self.warm_tts(sess.voice))
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

    # ---------- warmup / cooldown ----------
    TTS_WARM_TTL_S = 60.0

    def _tts_needs_warm(self, voice: str) -> bool:
        return bool(self.s.tts_url) and time.time() - self._tts_warmed_at.get(voice, 0.0) > self.TTS_WARM_TTL_S

    async def warm_tts(self, voice: str) -> bool:
        """Synthesize a tiny probe with `voice` unless that succeeded within TTS_WARM_TTL_S. Never raises."""
        if not self.s.tts_url:
            return False
        if not self._tts_needs_warm(voice):
            return True
        fut = self._tts_warming.get(voice)
        if fut is None:                   # one probe per voice at a time; concurrent callers share it
            fut = self._tts_warming[voice] = asyncio.ensure_future(self._probe_tts(voice))
            fut.add_done_callback(lambda _f, v=voice: self._tts_warming.pop(v, None))
        return await asyncio.shield(fut)

    async def _probe_tts(self, voice: str) -> bool:
        from .audio import AudioError
        from .tts import TtsError, synthesize

        own = self.http is None
        http = self.http
        if own:
            import aiohttp
            http = aiohttp.ClientSession()
        try:
            _, _, ms = await synthesize(http, self.s.tts_url, self.s.tts_model, voice, ".", self.s.tts_timeout_s)
            self._tts_warmed_at[voice] = time.time()
            log.info("warmed TTS voice %s in %d ms", voice, ms)
            return True
        except (TtsError, AudioError, asyncio.TimeoutError) as e:
            log.warning("could not warm TTS voice %s: %s", voice, e)
            return False
        except Exception:  # noqa: BLE001 - warming is best effort
            log.exception("could not warm TTS voice %s", voice)
            return False
        finally:
            if own:
                await http.close()

    async def warmup(self, avatar_id: Optional[str], voice: Optional[str]) -> dict:
        """Make sure the renderer and the TTS voice are loaded and warm, e.g. right before an interview."""
        self.require_ready()
        if not self.catalog:
            raise ServiceError(500, "no_avatars", "no avatars on the model volume")
        if avatar_id is not None:
            av = self.catalog.get(avatar_id)
            if av is None:
                raise ServiceError(404, "unknown_avatar", f"unknown avatar {avatar_id!r}; have {list(self.catalog)}")
            targets = [av]
        else:
            av = next(iter(self.catalog.values()))
            targets = [av] if self._warmed_avatars else list(self.catalog.values())
        voice = voice or av.voice
        tts = asyncio.ensure_future(self.warm_tts(voice))   # concurrently with the render

        busy = any(x.timeline.busy for x in self.sessions.values())   # a turn is rendering: it's warm anyway
        warmed: list[str] = []
        if not busy:
            loop = asyncio.get_running_loop()
            for a in targets:
                if self.on_demand or a.id not in self._warmed_avatars:
                    # on demand, renderer.warm() first (re)starts the child process if it was unloaded
                    await loop.run_in_executor(self.pool, self.renderer.warm, a)
                    self._warmed_avatars.add(a.id)
                    warmed.append(a.id)
        return {"ok": True, "status": self.state, "avatar": av.id, "voice": voice, "warmed_avatars": warmed,
                "renderer_busy": busy, "tts_warmed": await tts, "model": self._model_status()}

    async def cooldown(self, force: bool = False) -> dict:
        """Free the GPU now (on-demand mode) instead of waiting for the idle timeout."""
        if self.sessions and not force:
            return {"ok": True, "unloaded": False, "reason": "active_sessions", "sessions": len(self.sessions),
                    "model": self._model_status()}
        for sid in list(self.sessions):
            try:
                await self.close_session(sid)
            except ServiceError:
                pass                       # closed concurrently
        if not self.on_demand:
            return {"ok": True, "unloaded": False, "reason": "persistent_mode", "sessions": len(self.sessions),
                    "model": self._model_status()}
        if self.renderer is None or not hasattr(self.renderer, "unload"):
            return {"ok": True, "unloaded": False, "reason": "not_loaded", "sessions": len(self.sessions),
                    "model": self._model_status()}
        # in the render pool, so it queues behind a turn still winding down after close_session
        unloaded = await asyncio.get_running_loop().run_in_executor(self.pool, self.renderer.unload)
        self._warmed_avatars.clear()
        return {"ok": True, "unloaded": bool(unloaded), "sessions": len(self.sessions), "model": self._model_status()}

    # ---------- listening gestures ----------
    def set_state(self, sess: Session, state: object) -> dict:
        """Client behaviour state -> idle gesture (see gestures.ONE_SHOT / MODES). Never touches speech."""
        if not isinstance(state, str) or state not in STATES:
            raise ServiceError(400, "invalid_state", f"state must be one of {list(STATES)}")
        self.preload(sess)       # the client reports conversation state: an answer is likely coming
        return sess.director.set_state(state)

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
            if self.on_demand and not self.renderer.loaded:
                log.info("session %s: renderer unloaded, loading it for this turn", sess.id)
                loop.call_soon_threadsafe(self._send_event, sess, {"type": "model_loading"})
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
