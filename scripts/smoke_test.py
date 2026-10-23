#!/usr/bin/env python3
"""End-to-end smoke test against a running service, as a real WebRTC client (aiortc).

    python scripts/smoke_test.py --base http://localhost:8000 [--token T] [--avatar alexa] [--text "..."]

Checks: /health ready; session + offer/answer; ICE connected; idle video arrives at ~25 fps;
`say` -> utterance_start/utterance_end events; the voice is audible; video keeps ~25 fps while speaking;
audio and video durations match; a second session is refused while the first speaks (if max_speakers=1);
`stop` cancels; cleanup. Prints a JSON report; exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


class Client:
    def __init__(self, http, base, headers):
        self.http, self.base, self.h = http, base, headers
        self.pc = RTCPeerConnection()
        self.events, self.video_t, self.audio = [], [], []
        self.video_size = None
        self.tasks = []

    async def api(self, method, path, body=None, ok=(200, 201, 202)):
        async with self.http.request(method, self.base + path, json=body, headers=self.h) as r:
            j = await r.json(content_type=None)
            if r.status not in ok:
                raise RuntimeError(f"{method} {path} -> {r.status} {j}")
            return r.status, j

    async def connect(self, avatar):
        _, s = await self.api("POST", "/v1/sessions", {"avatar": avatar} if avatar else {})
        self.sid = s["session_id"]
        self.pc.addTransceiver("video", direction="recvonly")
        self.pc.addTransceiver("audio", direction="recvonly")
        ch = self.pc.createDataChannel("events")
        ch.on("message", lambda m: self.events.append({**json.loads(m), "t": time.time()}))

        @self.pc.on("track")
        def on_track(track):
            self.tasks.append(asyncio.ensure_future(self._pull(track)))

        await self.pc.setLocalDescription(await self.pc.createOffer())
        _, ans = await self.api("POST", f"/v1/sessions/{self.sid}/offer",
                                {"type": "offer", "sdp": self.pc.localDescription.sdp})
        await self.pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        for _ in range(100):   # both ends must be connected (the server's DTLS finishes a little later)
            if self.pc.connectionState == "connected":
                _, st = await self.api("GET", f"/v1/sessions/{self.sid}")
                if st["connection_state"] == "connected":
                    break
            await asyncio.sleep(0.1)
        return self.pc.connectionState

    async def _pull(self, track):
        try:
            while True:
                f = await track.recv()
                now = time.time()
                if track.kind == "video":
                    self.video_t.append(now)
                    self.video_size = (f.width, f.height)
                else:
                    a = f.to_ndarray().astype(np.float32).reshape(-1) / 32768.0
                    self.audio.append((now, float(np.sqrt(np.mean(a ** 2)))))
        except MediaStreamError:
            pass

    def fps(self, t0, t1):
        n = sum(1 for t in self.video_t if t0 <= t < t1)
        return n / max(1e-6, t1 - t0)

    def first_voice(self, after):
        for t, rms in self.audio:
            if t >= after and rms > 0.02:
                return t
        return None

    def voiced_seconds(self, t0, t1):
        return sum(0.02 for t, rms in self.audio if t0 <= t < t1 and rms > 0.01)

    async def close(self):
        try:
            await self.api("DELETE", f"/v1/sessions/{self.sid}")
        except Exception:  # noqa: BLE001
            pass
        await self.pc.close()
        for t in self.tasks:
            t.cancel()


async def main(a) -> int:
    headers = {"Authorization": f"Bearer {a.token}"} if a.token else {}
    report, checks = {}, {}
    async with aiohttp.ClientSession() as http:
        deadline = time.time() + a.wait_ready
        while True:
            async with http.get(a.base + "/health") as r:
                h = await r.json(content_type=None)
            if h.get("status") == "ready" or time.time() > deadline or h.get("status") == "error":
                break
            await asyncio.sleep(5)
        report["health"] = h
        checks["health_ready"] = h.get("status") == "ready"
        if not checks["health_ready"]:
            print(json.dumps({"checks": checks, "report": report}, indent=1))
            return 1

        c = Client(http, a.base, headers)
        try:
            checks["ice_connected"] = await c.connect(a.avatar) == "connected"
            await asyncio.sleep(3)
            now = time.time()
            report["idle_fps"] = round(c.fps(now - 2, now), 1)
            checks["idle_video_25fps"] = report["idle_fps"] >= 23
            report["video_size"] = c.video_size

            t_say = time.time()
            _, r = await c.api("POST", f"/v1/sessions/{c.sid}/say", {"text": a.text})
            report["say"] = r
            uid = r["utterance_id"]
            # a second session must be refused while this one speaks (max_speakers=1)
            if h.get("max_speakers") == 1 and h.get("max_sessions", 1) > 1:
                c2 = Client(http, a.base, headers)
                try:
                    await c2.connect(a.avatar)
                    st, j = await c2.api("POST", f"/v1/sessions/{c2.sid}/say", {"text": "Hello."}, ok=(202, 409))
                    checks["second_speaker_refused"] = st == 409 and j["error"]["code"] == "busy"
                finally:
                    await c2.close()
            end = None
            for _ in range(int(a.timeout * 10)):
                end = next((e for e in c.events if e["type"] == "utterance_end" and e["id"] == uid), None)
                if end:
                    break
                await asyncio.sleep(0.1)
            start = next((e for e in c.events if e["type"] == "utterance_start" and e["id"] == uid), None)
            checks["events"] = bool(start and end and not end.get("cancelled"))
            voice = c.first_voice(t_say)
            report["say_to_start_event_ms"] = round((start["t"] - t_say) * 1000) if start else None
            report["say_to_first_voice_ms"] = round((voice - t_say) * 1000) if voice else None
            report["say_to_first_word_ms"] = (report["say_to_first_voice_ms"] - r["lead_silence_ms"]) if voice else None
            checks["voice_heard"] = voice is not None
            if start and end:
                t0, t1 = start["t"], end["t"]
                report["speaking_s"] = round(t1 - t0, 2)
                report["speaking_fps"] = round(c.fps(t0, t1), 1)
                report["voiced_s"] = round(c.voiced_seconds(t0, t1 + 0.3), 2)
                checks["speaking_video_25fps"] = report["speaking_fps"] >= 23
                # audio and video share one timeline: utterance duration ~= its audio length
                checks["av_duration_match"] = abs(report["speaking_s"] * 1000 - r["audio_ms"]) < 400
            _, sess = await c.api("GET", f"/v1/sessions/{c.sid}")
            report["utterance_stats"] = sess["utterances"][-1] if sess["utterances"] else None

            # barge-in: start a long line and stop it
            _, r2 = await c.api("POST", f"/v1/sessions/{c.sid}/say", {"text": a.long_text})
            await asyncio.sleep(1.5)
            _, s = await c.api("POST", f"/v1/sessions/{c.sid}/stop")
            await asyncio.sleep(0.5)
            checks["stop_cancels"] = r2["utterance_id"] in s["cancelled"] and any(
                e["type"] == "utterance_end" and e["id"] == r2["utterance_id"] and e.get("cancelled") for e in c.events)
        finally:
            await c.close()
    ok = all(checks.values())
    print(json.dumps({"ok": ok, "checks": checks, "report": report}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--token", default="")
    p.add_argument("--avatar", default="")
    p.add_argument("--text", default="Thanks for joining today. Tell me about a time you had to make a difficult technical decision.")
    p.add_argument("--long-text", default="Let's move on to the system design question. How would you design a rate limiter "
                                          "for an API that serves millions of requests per minute across several regions?")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--wait-ready", type=float, default=0, help="seconds to wait for /health=ready")
    sys.exit(asyncio.run(main(p.parse_args())))
