"""Settings, all from environment variables (see README "Configuration")."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _list(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.environ.get("AVATAR_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("AVATAR_PORT", 8000))
    # read-only model dir: ditto/ (Hugging Face digital-avatar/ditto-talkinghead) + avatars/
    models_dir: str = field(default_factory=lambda: os.environ.get("AVATAR_MODELS_DIR", "/models"))
    # writable cache dir: GPU-specific TensorRT engines + pre-rendered idle loops
    data_dir: str = field(default_factory=lambda: os.environ.get("AVATAR_DATA_DIR", "/data"))
    ditto_dir: str = field(default_factory=lambda: os.environ.get("DITTO_DIR", "/opt/ditto"))
    # OpenAI-compatible TTS (POST {tts_url}/v1/audio/speech returning WAV), e.g. live-stt + Kokoro
    tts_url: str = field(default_factory=lambda: os.environ.get("AVATAR_TTS_URL", "http://localhost:8000").rstrip("/"))
    tts_model: str = field(default_factory=lambda: os.environ.get("AVATAR_TTS_MODEL", "kokoro-82m-v1"))
    tts_timeout_s: int = field(default_factory=lambda: _int("AVATAR_TTS_TIMEOUT_S", 30))
    # bearer token for /v1/*; empty = no auth (only for a private network)
    api_token: str = field(default_factory=lambda: os.environ.get("AVATAR_API_TOKEN", ""))
    cors_origins: list[str] = field(default_factory=lambda: _list("AVATAR_CORS_ORIGINS", "*"))
    max_sessions: int = field(default_factory=lambda: _int("AVATAR_MAX_SESSIONS", 4))
    # sessions allowed to be *speaking* at the same time (one RTX 3090 sustains one at >= 25 fps)
    max_speakers: int = field(default_factory=lambda: _int("AVATAR_MAX_SPEAKERS", 1))
    session_idle_timeout_s: int = field(default_factory=lambda: _int("AVATAR_SESSION_IDLE_TIMEOUT_S", 900))
    idle_loop_seconds: int = field(default_factory=lambda: _int("AVATAR_IDLE_LOOP_SECONDS", 8))
    # diffusion steps of Ditto's motion model (online config default: 10)
    sampling_steps: int = field(default_factory=lambda: _int("AVATAR_SAMPLING_STEPS", 10))
    max_text_chars: int = field(default_factory=lambda: _int("AVATAR_MAX_TEXT_CHARS", 1000))
    max_audio_seconds: int = field(default_factory=lambda: _int("AVATAR_MAX_AUDIO_SECONDS", 30))
    # optional STUN/TURN for clients outside the LAN, e.g. "stun:stun.l.google.com:19302"
    ice_servers: list[str] = field(default_factory=lambda: _list("AVATAR_ICE_SERVERS"))
    # skip GPU work entirely (API / unit tests only)
    fake_renderer: bool = field(default_factory=lambda: os.environ.get("AVATAR_FAKE_RENDERER", "") == "1")

    @property
    def ditto_models(self) -> str:
        return os.path.join(self.models_dir, "ditto")

    @property
    def avatars_dir(self) -> str:
        return os.path.join(self.models_dir, "avatars")
