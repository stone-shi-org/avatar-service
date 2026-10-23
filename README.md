# avatar-service

A real-time, photoreal, lip-synced talking head served over **WebRTC**. Send text (or audio). The service speaks it with an
OpenAI-compatible TTS, animates a single portrait photo with
[Ditto](https://github.com/antgroup/ditto-talkinghead) (TensorRT), and streams video and voice to the browser in one
lip-synced `MediaStream`.

It was built as the "photoreal interviewer" for a mock-interview app, but it knows nothing about interviews.

- **One photo per avatar.** No video of the person is needed. Idle motion (blinks, head sway) is pre-rendered once.
- **Measured on an RTX 3090** with an STT and TTS model on the same card:
  - 25–30 fps at 1280×720;
  - about 330 ms (p95 about 400 ms) from audio to stall-free video;
  - about 3 GB of VRAM.
- **One speaking session per GPU.** A second session asking to speak gets `409 busy`, so clients can fall back to
  something cheaper. Any number of connected but silent sessions is fine (idle frames cost no GPU).
- **Engines rebuilt per GPU.** Ditto's shipped "Ampere+" engines are about 20 % slower. On first start the service
  rebuilds the two per-frame hot spots natively for your GPU, taking a few minutes, and caches them.
- **Models stay out of the image.** They live on a mounted volume.

## Quick start

```bash
./build.sh                                   # image: avatar:latest (~18 GB, CUDA 12.1 + TensorRT 8.6)
scripts/download_models.sh ./models          # ~4.5 GB from Hugging Face (digital-avatar/ditto-talkinghead)
cp my-portrait.png ./models/avatars/alexa.png
cp docker-compose.example.yml docker-compose.yml && docker compose up -d
docker logs -f avatar                        # first start: "building native decoder engine ..." (~5-10 min)
python scripts/smoke_test.py --base http://localhost:4040 --wait-ready 900
```

Open `http://<host>:4040/` for the built-in test page (Connect → Say).

## Avatars

Put portraits in `<models>/avatars/`. Use a head-and-shoulders shot facing the camera, mouth closed, even lighting,
and **even width and height**; 1280×720 works well. Optionally describe them in `avatars.json`:

```json
{
  "alexa":  {"image": "alexa.png",  "name": "Alexa",  "voice": "af_heart",   "emo": 4},
  "marcus": {"image": "marcus.png", "name": "Marcus", "voice": "am_michael", "emo": 4}
}
```

`voice` is passed to the TTS. `emo` is Ditto's emotion class (0–7; 4 is the model's default). The idle loop for each
avatar is rendered at startup and cached in `<data>/idle/`.

## API

Every `/v1/*` call needs `Authorization: Bearer <AVATAR_API_TOKEN>` when that variable is set. Errors look like
`{"error": {"code", "message"}}`.

| Method & path | Body | Result |
|---|---|---|
| `GET /health` | | `200` when ready, `503` while starting (`status`: `building_engines`, `loading`, `warming_up`, `error`) |
| `GET /v1/avatars` | | `{avatars: [{id, name, voice, emo, image}]}` |
| `POST /v1/sessions` | `{avatar?, voice?}` | `201 {session_id, avatar, voice, frame_size, ice_servers}`; `429` if `AVATAR_MAX_SESSIONS` is reached |
| `POST /v1/sessions/{id}/offer` | `{type: "offer", sdp}` | SDP answer. The offer must receive one audio and one video track (`recvonly`), plus optionally a data channel |
| `POST /v1/sessions/{id}/say` | `{text}` or `{audio_wav_base64}`, `voice?`, `emo?` | `202 {utterance_id, tts_ms, audio_ms, lead_silence_ms}`; `409 busy` if another session is speaking; `409 not_connected` |
| `POST /v1/sessions/{id}/stop` | | barge-in: cancels everything queued or playing → `{cancelled: [ids]}` |
| `GET /v1/sessions/{id}` | | state + per-utterance timings |
| `DELETE /v1/sessions/{id}` | | close |

**Sentences** posted one after another are queued and play back to back.

**Data channel events.** If the offer creates a data channel (any label), the service sends JSON events:

- `{"type": "utterance_start", "id", "text", "start_ms", "lead_silence_ms", "audio_ms"}` when the voice for an utterance starts;
- `{"type": "utterance_end", "id", "cancelled"}` when it has finished (or was stopped).

Use these for captions and turn-taking.

**Sync.** Audio for an utterance is released once 8 frames are pre-rendered (`start_ms` on the shared media clock).
The video track then always shows the frame that belongs to the audio position: late frames are dropped and missing
ones hold the previous frame, so lips never drift behind the voice. Measured skew in Chrome was −34…+60 ms.

### Browser notes

- Chrome only feeds a remote WebRTC audio track into Web Audio (`createMediaStreamSource`, e.g. as an echo-canceller
  reference) while the track is also attached to a media element. Attach it to a muted `<audio>` element.
- `network_mode: host` (or a TURN server, via `AVATAR_ICE_SERVERS`) is needed for media to reach browsers. Bridge
  networking advertises container-internal ICE candidates.

## Configuration

| Variable | Default | |
|---|---|---|
| `AVATAR_PORT` | `8000` | HTTP port |
| `AVATAR_TTS_URL` | `http://localhost:8000` | OpenAI-compatible TTS base URL (`POST /v1/audio/speech`, WAV) |
| `AVATAR_TTS_MODEL` | `kokoro-82m-v1` | |
| `AVATAR_API_TOKEN` | *(empty = open)* | bearer token for `/v1/*` |
| `AVATAR_CORS_ORIGINS` | `*` | comma-separated |
| `AVATAR_MAX_SESSIONS` | `4` | connected sessions |
| `AVATAR_MAX_SPEAKERS` | `1` | sessions speaking at once (one RTX 3090 sustains one at 25 fps) |
| `AVATAR_SESSION_IDLE_TIMEOUT_S` | `900` | sessions without API calls are closed |
| `AVATAR_IDLE_LOOP_SECONDS` | `8` | length of the pre-rendered idle loop (played ping-pong) |
| `AVATAR_SAMPLING_STEPS` | `10` | Ditto motion-diffusion steps (5 is a little faster, with more startup jitter) |
| `AVATAR_NATIVE_ENGINES` | `1` | `0` = use the shipped Ampere+ engines as they are (slower) |
| `AVATAR_ICE_SERVERS` | | e.g. `stun:stun.l.google.com:19302,turn:...` |
| `AVATAR_MAX_TEXT_CHARS` / `AVATAR_MAX_AUDIO_SECONDS` | `1000` / `30` | per utterance |
| `AVATAR_MODELS_DIR` / `AVATAR_DATA_DIR` | `/models` / `/data` | |

## Development

```bash
./build.sh --test                 # build, then run the GPU-free tests inside the image (JUnit in test-reports/)
```

The tests use a fake renderer and a fake TTS, and drive the real server with `scripts/smoke_test.py` as a WebRTC
client (aiortc). They need no GPU and no models.

Layout:

- `avatar_service/session.py`: the audio/video timeline (pure Python, unit-tested);
- `media.py`: aiortc tracks;
- `ditto_engine.py`: the Ditto wrapper;
- `engines.py`: the native TensorRT builds;
- `service.py` and `server.py`: sessions and the API.

## Licenses

- **This repository:** Apache-2.0.
- **Ditto:** code and weights are Apache-2.0 (antgroup/ditto-talkinghead). Its face-detection and landmark models
  appear to be derived from InsightFace, whose pretrained models are licensed for non-commercial research only. They
  run only when an avatar is registered. Check this before commercial use.
- **Portraits:** only use photos you have the right to animate. AI-generated faces are the safest choice.
