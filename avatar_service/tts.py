"""OpenAI-compatible TTS client (`POST /v1/audio/speech`, WAV out), e.g. live-stt serving Kokoro."""
from __future__ import annotations

import time

import aiohttp
import numpy as np

from .audio import decode_wav


class TtsError(RuntimeError):
    pass


async def synthesize(session: aiohttp.ClientSession, url: str, model: str, voice: str, text: str,
                     timeout_s: int = 30) -> tuple[np.ndarray, int, int]:
    """Returns (float32 mono audio, sample rate, elapsed ms)."""
    t = time.time()
    body = {"model": model, "voice": voice, "input": text, "response_format": "wav"}
    try:
        async with session.post(f"{url}/v1/audio/speech", json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            data = await r.read()
            if r.status != 200:
                raise TtsError(f"TTS HTTP {r.status}: {data[:300].decode('utf-8', 'replace')}")
    except aiohttp.ClientError as e:
        raise TtsError(f"TTS unreachable: {e}") from e
    x, sr = decode_wav(data)
    return x, sr, round((time.time() - t) * 1000)
