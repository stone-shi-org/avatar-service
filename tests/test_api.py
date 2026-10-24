"""The real server (fake GPU renderer + fake TTS) driven by scripts/smoke_test.py as a WebRTC client."""
import asyncio
import importlib.util
import os
import socket

import cv2
import numpy as np
from aiohttp import web

from avatar_service.audio import encode_wav
from avatar_service.config import Settings
from avatar_service.server import build_app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("smoke_test", os.path.join(ROOT, "scripts", "smoke_test.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def fake_tts(request):
    body = await request.json()
    sec = min(3.0, 0.4 + 0.05 * len(body["input"].split()))
    t = np.arange(int(24000 * sec)) / 24000
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    x[:7200] = 0  # 300 ms leading silence, like Kokoro
    return web.Response(body=encode_wav(x, 24000), content_type="audio/wav")


async def run(tmp_path, token, unload_s=0):
    av = tmp_path / "models" / "avatars"
    av.mkdir(parents=True)
    cv2.imwrite(str(av / "alexa.png"), np.full((360, 640, 3), 120, np.uint8))
    tts_port, port = free_port(), free_port()
    tts_app = web.Application()
    tts_app.add_routes([web.post("/v1/audio/speech", fake_tts)])
    tts_runner = web.AppRunner(tts_app)
    await tts_runner.setup()
    await web.TCPSite(tts_runner, "127.0.0.1", tts_port).start()

    s = Settings()
    s.models_dir, s.data_dir = str(tmp_path / "models"), str(tmp_path / "data")
    s.tts_url, s.fake_renderer, s.api_token, s.max_sessions = f"http://127.0.0.1:{tts_port}", True, token, 3
    s.model_idle_unload_s = unload_s
    runner = web.AppRunner(build_app(s))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        args = type("A", (), dict(base=f"http://127.0.0.1:{port}", token=token, avatar="alexa", timeout=20,
                                  wait_ready=20, text="Thanks for joining today, let's get started.",
                                  long_text=" ".join(["word"] * 40)))
        return await smoke.main(args)
    finally:
        await runner.cleanup()
        await tts_runner.cleanup()


def test_end_to_end_with_fake_renderer(tmp_path, capsys):
    rc = asyncio.run(run(tmp_path, token="t0ken"))
    out = capsys.readouterr().out
    assert rc == 0, out


def test_end_to_end_with_on_demand_renderer(tmp_path, capsys):
    """Same, with the (fake) renderer in a child process that is stopped after 1 s idle and restarted per turn."""
    rc = asyncio.run(run(tmp_path, token="t0ken", unload_s=1))
    out = capsys.readouterr().out
    assert rc == 0, out


def test_auth_required(tmp_path):
    async def go():
        s = Settings()
        s.models_dir, s.data_dir, s.fake_renderer, s.api_token = str(tmp_path), str(tmp_path), True, "secret"
        port = free_port()
        runner = web.AppRunner(build_app(s))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        import aiohttp
        try:
            async with aiohttp.ClientSession() as h:
                async with h.get(f"http://127.0.0.1:{port}/v1/avatars") as r:
                    assert r.status == 401
                async with h.get(f"http://127.0.0.1:{port}/v1/avatars", headers={"Authorization": "Bearer secret"}) as r:
                    assert r.status == 200
                async with h.get(f"http://127.0.0.1:{port}/health") as r:   # health stays open
                    assert r.status in (200, 503)
        finally:
            await runner.cleanup()
    asyncio.run(go())


def test_state_endpoint(tmp_path):
    async def go():
        import aiohttp
        av = tmp_path / "models" / "avatars"
        av.mkdir(parents=True)
        cv2.imwrite(str(av / "alexa.png"), np.full((360, 640, 3), 120, np.uint8))
        s = Settings()
        s.models_dir, s.data_dir, s.fake_renderer, s.api_token = str(tmp_path / "models"), str(tmp_path / "data"), True, "k"
        port = free_port()
        runner = web.AppRunner(build_app(s))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        base, auth = f"http://127.0.0.1:{port}", {"Authorization": "Bearer k"}
        try:
            async with aiohttp.ClientSession() as h:
                for _ in range(100):
                    async with h.get(f"{base}/health") as r:
                        if r.status == 200:
                            break
                    await asyncio.sleep(0.1)
                async with h.post(f"{base}/v1/sessions", json={}, headers=auth) as r:
                    sid = (await r.json())["session_id"]
                url = f"{base}/v1/sessions/{sid}/state"
                async with h.post(url, json={"state": "listening"}) as r:
                    assert r.status == 401
                async with h.post(f"{base}/v1/sessions/ses_nope/state", json={"state": "listening"}, headers=auth) as r:
                    assert r.status == 404
                for bad in ({"state": "dancing"}, {"state": 3}, {}):
                    async with h.post(url, json=bad, headers=auth) as r:
                        assert r.status == 400 and (await r.json())["error"]["code"] == "invalid_state"
                async with h.post(url, data="nope", headers=auth) as r:
                    assert r.status == 400
                expect = {"listening": (None, "applied"), "backchannel": ("nod", "pending"), "probe": ("tilt", "pending"),
                          "hint": ("smile", "pending"), "wrap_up": ("smile", "pending"),
                          "silent_thinking": (None, "applied"), "paused": (None, "applied")}
                for state, (clip, status) in expect.items():
                    async with h.post(url, json={"state": state}, headers=auth) as r:
                        body = await r.json()
                        assert r.status == 200, body
                        assert body["state"] == state and body["clip"] == clip and body["status"] == status
                async with h.get(f"{base}/v1/sessions/{sid}", headers=auth) as r:
                    g = (await r.json())["gesture"]
                    assert g["mode"] == "paused" and g["clips"] == ["glance_down", "nod", "smile", "tilt"]
        finally:
            await runner.cleanup()
    asyncio.run(go())


# ---------- POST /v1/warmup and /v1/cooldown ----------
async def _warm_server(tmp_path, unload_s=0, tts=True):
    """Fake renderer + counting fake TTS (or an unreachable TTS URL). Returns (base, auth, tts_calls, cleanup)."""
    av = tmp_path / "models" / "avatars"
    av.mkdir(parents=True)
    for name in ("alexa", "brian"):
        cv2.imwrite(str(av / f"{name}.png"), np.full((360, 640, 3), 120, np.uint8))
    calls = []
    runners = []
    tts_port, port = free_port(), free_port()
    if tts:
        async def counting_tts(request):
            body = await request.json()
            calls.append(body)
            return await fake_tts_body(body)
        tts_app = web.Application()
        tts_app.add_routes([web.post("/v1/audio/speech", counting_tts)])
        r = web.AppRunner(tts_app)
        await r.setup()
        await web.TCPSite(r, "127.0.0.1", tts_port).start()
        runners.append(r)
    s = Settings()
    s.models_dir, s.data_dir = str(tmp_path / "models"), str(tmp_path / "data")
    s.tts_url, s.fake_renderer, s.api_token = f"http://127.0.0.1:{tts_port}", True, "k"
    s.model_idle_unload_s, s.tts_timeout_s = unload_s, 5
    app = build_app(s)
    r = web.AppRunner(app)
    await r.setup()
    await web.TCPSite(r, "127.0.0.1", port).start()
    runners.insert(0, r)

    async def cleanup():
        for x in runners:
            await x.cleanup()
    return f"http://127.0.0.1:{port}", {"Authorization": "Bearer k"}, calls, cleanup


async def fake_tts_body(body):
    t = np.arange(2400) / 24000
    return web.Response(body=encode_wav((0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 24000),
                        content_type="audio/wav")


async def _wait_ready(h, base, timeout=60):
    for _ in range(int(timeout * 10)):
        async with h.get(f"{base}/health") as r:
            if r.status == 200:
                return await r.json()
        await asyncio.sleep(0.1)
    raise AssertionError("service never became ready")


def test_warmup_cooldown_persistent(tmp_path):
    async def go():
        import aiohttp
        base, auth, calls, cleanup = await _warm_server(tmp_path)
        try:
            async with aiohttp.ClientSession() as h:
                async with h.post(f"{base}/v1/warmup") as r:
                    assert r.status == 401
                await _wait_ready(h, base)
                startup_probes = len(calls)     # startup warms the catalog's voices
                # no body at all; startup already rendered every avatar, so nothing to warm again
                async with h.post(f"{base}/v1/warmup", headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200, b
                    assert b["ok"] and b["status"] == "ready" and b["avatar"] == "alexa" and b["voice"] == "af_heart"
                    assert b["warmed_avatars"] == [] and b["tts_warmed"] is True
                    assert b["model"]["state"] == "loaded"
                assert len(calls) == startup_probes  # voice probed < 60 s ago: no new TTS request
                async with h.post(f"{base}/v1/warmup", json={"avatar": "brian", "voice": "am_adam"}, headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200 and b["avatar"] == "brian" and b["voice"] == "am_adam" and b["tts_warmed"]
                assert calls[-1]["voice"] == "am_adam" and calls[-1]["input"] == "."
                n = len(calls)
                async with h.post(f"{base}/v1/warmup", json={"voice": "am_adam"}, headers=auth) as r:
                    assert r.status == 200
                assert len(calls) == n               # cached for 60 s
                async with h.post(f"{base}/v1/warmup", json={"avatar": "nobody"}, headers=auth) as r:
                    assert r.status == 404 and (await r.json())["error"]["code"] == "unknown_avatar"
                async with h.post(f"{base}/v1/warmup", json={"avatar": 3}, headers=auth) as r:
                    assert r.status == 400
                async with h.post(f"{base}/v1/warmup", data="nope", headers=auth) as r:
                    assert r.status == 400

                async with h.post(f"{base}/v1/sessions", json={}, headers=auth) as r:
                    sid = (await r.json())["session_id"]
                async with h.post(f"{base}/v1/cooldown", headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200 and b == {"ok": True, "unloaded": False, "reason": "active_sessions",
                                                     "sessions": 1, "model": b["model"]}
                async with h.post(f"{base}/v1/cooldown", json={"force": "yes"}, headers=auth) as r:
                    assert r.status == 400
                async with h.post(f"{base}/v1/cooldown", json={"force": True}, headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200 and not b["unloaded"] and b["reason"] == "persistent_mode"
                    assert b["sessions"] == 0 and b["model"]["state"] == "loaded"
                async with h.get(f"{base}/v1/sessions/{sid}", headers=auth) as r:
                    assert r.status == 404
        finally:
            await cleanup()
    asyncio.run(go())


def test_warmup_cooldown_on_demand(tmp_path):
    async def go():
        import aiohttp
        base, auth, calls, cleanup = await _warm_server(tmp_path, unload_s=300)
        try:
            async with aiohttp.ClientSession() as h:
                hl = await _wait_ready(h, base)
                assert hl["model"]["state"] == "loaded"
                loads = hl["model"]["loads"]
                async with h.post(f"{base}/v1/cooldown", json={}, headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200 and b["unloaded"] is True and b["model"]["state"] == "unloaded", b
                async with h.post(f"{base}/v1/cooldown", headers=auth) as r:   # already unloaded
                    b = await r.json()
                    assert r.status == 200 and b["unloaded"] is False and b["model"]["state"] == "unloaded"
                # re-warm: starts the renderer process again and renders every avatar (none warm after unload)
                async with h.post(f"{base}/v1/warmup", json={}, headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200, b
                    assert b["model"]["state"] == "loaded" and b["model"]["loads"] == loads + 1
                    assert sorted(b["warmed_avatars"]) == ["alexa", "brian"]
                async with h.post(f"{base}/v1/warmup", json={"avatar": "brian"}, headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200 and b["warmed_avatars"] == ["brian"] and b["model"]["loads"] == loads + 1

                async with h.post(f"{base}/v1/sessions", json={}, headers=auth) as r:
                    assert r.status == 201
                async with h.post(f"{base}/v1/cooldown", json={"force": False}, headers=auth) as r:
                    b = await r.json()
                    assert b["unloaded"] is False and b["reason"] == "active_sessions" and b["sessions"] == 1
                    assert b["model"]["state"] == "loaded"
                async with h.post(f"{base}/v1/cooldown", json={"force": True}, headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200 and b["unloaded"] is True and b["sessions"] == 0
                    assert b["model"]["state"] == "unloaded"
        finally:
            await cleanup()
    asyncio.run(go())


def test_warmup_survives_unreachable_tts(tmp_path):
    async def go():
        import aiohttp
        base, auth, _, cleanup = await _warm_server(tmp_path, tts=False)
        try:
            async with aiohttp.ClientSession() as h:
                await _wait_ready(h, base)
                async with h.post(f"{base}/v1/warmup", json={"avatar": "alexa"}, headers=auth) as r:
                    b = await r.json()
                    assert r.status == 200 and b["ok"] and b["tts_warmed"] is False
        finally:
            await cleanup()
    asyncio.run(go())


def test_create_session_wakes_tts(tmp_path):
    async def go():
        import aiohttp
        base, auth, calls, cleanup = await _warm_server(tmp_path)
        try:
            async with aiohttp.ClientSession() as h:
                await _wait_ready(h, base)
                n = len(calls)
                async with h.post(f"{base}/v1/sessions", json={"voice": "bf_emma"}, headers=auth) as r:
                    assert r.status == 201
                for _ in range(50):
                    if len(calls) > n:
                        break
                    await asyncio.sleep(0.1)
                assert [c["voice"] for c in calls[n:]] == ["bf_emma"]
                async with h.post(f"{base}/v1/sessions", json={"voice": "bf_emma"}, headers=auth) as r:
                    assert r.status == 201
                await asyncio.sleep(0.3)
                assert len(calls) == n + 1        # warmed < 60 s ago: no second probe
        finally:
            await cleanup()
    asyncio.run(go())
