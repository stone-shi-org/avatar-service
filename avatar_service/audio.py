"""Pure audio helpers (numpy only, no GPU): WAV decoding, resampling, leading-silence detection."""
from __future__ import annotations

import io
import struct

import numpy as np

FPS = 25
OUT_SR = 48000           # WebRTC audio track
DITTO_SR = 16000         # Ditto's HuBERT input
SAMPLES_PER_FRAME = OUT_SR // FPS  # 1920


class AudioError(ValueError):
    pass


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV to mono float32 in [-1, 1].

    Tolerates streaming headers whose RIFF/data sizes are 0xFFFFFFFF (Kokoro via live-stt sends those):
    the data chunk then simply runs to the end of the buffer.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[body:body + 16])
        elif cid == b"data":
            end = len(data) if size in (0, 0xFFFFFFFF) or body + size > len(data) else body + size
            pcm = data[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or pcm is None:
        raise AudioError("WAV without fmt/data chunk")
    tag, channels, sr, _, _, bits = fmt
    if tag not in (1, 0xFFFE) or bits not in (16, 32) or channels < 1:
        raise AudioError(f"unsupported WAV format tag={tag} bits={bits}")
    width = bits // 8
    pcm = pcm[: len(pcm) // (width * channels) * (width * channels)]
    if bits == 16:
        x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
    else:
        x = np.frombuffer(pcm, "<i4").astype(np.float32) / 2147483648.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, sr


def encode_wav(x: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    b = io.BytesIO()
    b.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE")
    b.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    b.write(b"data" + struct.pack("<I", len(pcm)) + pcm)
    return b.getvalue()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler; plenty for speech going into HuBERT / Opus."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def lead_silence_ms(x: np.ndarray, sr: int, threshold: float = 0.02, window: int = 512) -> int:
    """Milliseconds before the RMS over `window` samples first exceeds `threshold`."""
    if len(x) < window:
        return 0
    rms = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(window) / window, mode="valid"))
    hit = np.flatnonzero(rms > threshold)
    return int(round(hit[0] * 1000 / sr)) if len(hit) else 0


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def pad_to_frames(x: np.ndarray, frames: int, samples_per_frame: int = SAMPLES_PER_FRAME) -> np.ndarray:
    """Pad (silence) or trim so the audio lasts exactly `frames` video frames."""
    want = frames * samples_per_frame
    if len(x) >= want:
        return x[:want]
    return np.concatenate([x, np.zeros(want - len(x), dtype=x.dtype)])


def frames_for(seconds: float) -> int:
    return int(np.ceil(seconds * FPS))
