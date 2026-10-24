"""HTTP API (aiohttp). See README "API"."""
from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os

import aiohttp
from aiohttp import web

from . import __version__
from .audio import AudioError, decode_wav
from .config import Settings
from .service import AvatarService, ServiceError
from .tts import TtsError, synthesize

log = logging.getLogger("avatar.server")
STATIC = os.path.join(os.path.dirname(__file__), "static")
SVC = web.AppKey("svc", AvatarService)
HTTP = web.AppKey("http", aiohttp.ClientSession)
CFG = web.AppKey("cfg", Settings)


def _err(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


@web.middleware
async def middleware(request: web.Request, handler):
    s: Settings = request.app[CFG]
    origin = request.headers.get("Origin")
    allow = None
    if origin and ("*" in s.cors_origins or origin in s.cors_origins):
        allow = "*" if "*" in s.cors_origins else origin
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    elif request.path.startswith("/v1/") and s.api_token and not hmac.compare_digest(
            request.headers.get("Authorization", ""), f"Bearer {s.api_token}"):
        resp = _err(401, "unauthorized", "missing or wrong bearer token")
    else:
        try:
            resp = await handler(request)
        except ServiceError as e:
            resp = _err(e.status, e.code, e.message)
        except web.HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("unhandled error on %s", request.path)
            resp = _err(500, "internal_error", f"{type(e).__name__}")
    if allow:
        resp.headers["Access-Control-Allow-Origin"] = allow
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return resp


async def _json(request: web.Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise ServiceError(400, "invalid_json", "request body must be JSON")
    if not isinstance(body, dict):
        raise ServiceError(400, "invalid_json", "request body must be a JSON object")
    return body


async def index(request):
    return web.FileResponse(os.path.join(STATIC, "index.html"))


async def health(request):
    h = request.app[SVC].health()
    h["version"] = __version__
    return web.json_response(h, status=200 if h["status"] == "ready" else 503)


async def avatars(request):
    return web.json_response({"avatars": [a.public() for a in request.app[SVC].catalog.values()]})


async def create_session(request):
    svc = request.app[SVC]
    body = await _json(request) if request.can_read_body else {}
    sess = svc.create_session(body.get("avatar"), body.get("voice"))
    return web.json_response({**sess.public(), "ice_servers": svc.s.ice_servers}, status=201)


async def get_session(request):
    return web.json_response(request.app[SVC].get(request.match_info["sid"]).public())


async def delete_session(request):
    await request.app[SVC].close_session(request.match_info["sid"])
    return web.json_response({"ok": True})


async def offer(request):
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

    from .media import AvatarAudioTrack, AvatarVideoTrack, MediaClock

    svc = request.app[SVC]
    sess = svc.get(request.match_info["sid"])
    body = await _json(request)
    if body.get("type") != "offer" or not isinstance(body.get("sdp"), str):
        raise ServiceError(400, "invalid_offer", "need {type: 'offer', sdp}")
    if sess.pc is not None:
        raise ServiceError(409, "already_connected", "session already has a peer connection; create a new session")
    cfg = RTCConfiguration([RTCIceServer(u) for u in svc.s.ice_servers]) if svc.s.ice_servers else None
    pc = RTCPeerConnection(cfg)
    sess.pc = pc
    clock = MediaClock()
    video = AvatarVideoTrack(clock, sess.timeline, sess.director.frame, sess.portrait)
    sess.video_track = video

    @pc.on("datachannel")
    def on_channel(ch):
        sess.channel = ch

    @pc.on("connectionstatechange")
    async def on_state():
        sess.connection_state = pc.connectionState
        log.info("session %s: %s", sess.id, pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            svc.stop(sess)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
    # attach after the offer so each track lands on the client's matching recvonly transceiver;
    # both get the peer connection's single msid stream -> the browser lip-syncs them
    kinds = {t.kind for t in pc.getTransceivers()}
    if not {"audio", "video"} <= kinds:
        await pc.close()
        sess.pc = None
        raise ServiceError(400, "invalid_offer", "offer must receive one audio and one video track")
    pc.addTrack(AvatarAudioTrack(clock, sess.timeline))
    pc.addTrack(video)
    await pc.setLocalDescription(await pc.createAnswer())
    return web.json_response({"type": pc.localDescription.type, "sdp": pc.localDescription.sdp})


async def say(request):
    svc = request.app[SVC]
    svc.require_ready()
    sess = svc.get(request.match_info["sid"])
    if sess.connection_state != "connected":
        raise ServiceError(409, "not_connected", f"peer connection is {sess.connection_state}")
    body = await _json(request)
    text = body.get("text") or ""
    emo = body.get("emo")
    if emo is not None and (not isinstance(emo, int) or not 0 <= emo <= 7):
        raise ServiceError(400, "invalid_emo", "emo must be an integer 0..7")
    svc.admit(sess)
    if body.get("audio_wav_base64"):
        try:
            audio, sr = decode_wav(base64.b64decode(body["audio_wav_base64"], validate=True))
        except (binascii.Error, AudioError) as e:
            raise ServiceError(400, "invalid_audio", str(e))
        tts_ms = 0
    else:
        if not isinstance(text, str) or not text.strip():
            raise ServiceError(400, "missing_text", "need text or audio_wav_base64")
        if len(text) > svc.s.max_text_chars:
            raise ServiceError(413, "text_too_long", f"at most {svc.s.max_text_chars} characters")
        try:
            audio, sr, tts_ms = await synthesize(request.app[HTTP], svc.s.tts_url, svc.s.tts_model,
                                                 body.get("voice") or sess.voice, text, svc.s.tts_timeout_s)
        except (TtsError, AudioError) as e:
            raise ServiceError(502, "tts_failed", str(e))
    svc.admit(sess)  # re-check: another session may have started while TTS ran
    utt = svc.speak(sess, audio, sr, text, tts_ms, emo)
    return web.json_response({"utterance_id": utt.id, "tts_ms": tts_ms, "audio_ms": round(len(utt.audio48) / 48),
                              "lead_silence_ms": utt.lead_silence_ms}, status=202)


async def set_state(request):
    svc = request.app[SVC]
    sess = svc.get(request.match_info["sid"])
    body = await _json(request)
    return web.json_response(svc.set_state(sess, body.get("state")))


async def stop(request):
    svc = request.app[SVC]
    return web.json_response({"cancelled": svc.stop(svc.get(request.match_info["sid"]))})


def build_app(settings: Settings | None = None, service: AvatarService | None = None) -> web.Application:
    s = settings or Settings()
    app = web.Application(middlewares=[middleware], client_max_size=16 * 1024 * 1024)
    app[CFG] = s
    app[SVC] = service or AvatarService(s)

    async def on_startup(app):
        import asyncio
        app[HTTP] = aiohttp.ClientSession()
        app[SVC].start(asyncio.get_running_loop())

    async def on_cleanup(app):
        for sid in list(app[SVC].sessions):
            await app[SVC].close_session(sid)
        app[SVC].shutdown()
        await app[HTTP].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes([
        web.get("/", index), web.get("/health", health), web.get("/v1/avatars", avatars),
        web.post("/v1/sessions", create_session), web.get("/v1/sessions/{sid}", get_session),
        web.delete("/v1/sessions/{sid}", delete_session), web.post("/v1/sessions/{sid}/offer", offer),
        web.post("/v1/sessions/{sid}/say", say), web.post("/v1/sessions/{sid}/stop", stop),
        web.post("/v1/sessions/{sid}/state", set_state),
    ])
    return app
