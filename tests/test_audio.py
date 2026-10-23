import struct

import numpy as np
import pytest

from avatar_service.audio import (AudioError, decode_wav, encode_wav, frames_for, lead_silence_ms, pad_to_frames,
                                  resample)


def tone(sr=24000, sec=1.0, lead=0.3):
    t = np.arange(int(sr * sec)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    x[: int(lead * sr)] = 0
    return x


def test_wav_roundtrip():
    x = tone()
    y, sr = decode_wav(encode_wav(x, 24000))
    assert sr == 24000 and len(y) == len(x)
    assert np.max(np.abs(y - x)) < 1e-3


def test_streaming_header_sizes_are_tolerated():
    data = bytearray(encode_wav(tone(), 24000))
    data[4:8] = struct.pack("<I", 0xFFFFFFFF)     # RIFF size
    data[40:44] = struct.pack("<I", 0xFFFFFFFF)   # data size (what Kokoro via live-stt sends)
    y, sr = decode_wav(bytes(data))
    assert sr == 24000 and len(y) == 24000


def test_rejects_non_wav():
    with pytest.raises(AudioError):
        decode_wav(b"ID3 not a wav at all")


def test_resample_length():
    assert len(resample(np.zeros(24000, np.float32), 24000, 48000)) == 48000
    assert len(resample(np.zeros(24000, np.float32), 24000, 16000)) == 16000


def test_lead_silence():
    assert abs(lead_silence_ms(resample(tone(lead=0.3), 24000, 48000), 48000) - 300) <= 15


def test_pad_to_frames():
    assert len(pad_to_frames(np.zeros(100, np.int16), 2)) == 3840
    assert len(pad_to_frames(np.zeros(5000, np.int16), 2)) == 3840
    assert frames_for(1.0) == 25 and frames_for(1.01) == 26
