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


async def run(tmp_path, token):
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
