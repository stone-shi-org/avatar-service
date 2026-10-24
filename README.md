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
  - about 3 GB of VRAM, which can be freed while nobody is speaking (see [Freeing the GPU when idle](#freeing-the-gpu-when-idle)).
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

`voice` is passed to the TTS. `emo` is Ditto's emotion class (0–7; 4 is the model's default). The idle loop and the
listening gestures for each avatar are rendered at startup and cached in `<data>/idle/`.

**Idle mouth.** Ditto animates a portrait from audio, and even near-silence makes its motion model predict small lip
movements, so the face looked about to speak. In the idle loop and the gesture clips the lips are held at the
portrait's own (closed) mouth; blinks and head sway are kept. Speech is not affected. This is why portraits must have
the mouth closed.

## API

Every `/v1/*` call needs `Authorization: Bearer <AVATAR_API_TOKEN>` when that variable is set. Errors look like
`{"error": {"code", "message"}}`.

| Method & path | Body | Result |
|---|---|---|
| `GET /health` | | `200` when ready, `503` while starting (`status`: `building_engines`, `loading`, `warming_up`, `error`); `model.state`: `loaded`, `unloaded`, `loading` |
| `GET /v1/avatars` | | `{avatars: [{id, name, voice, emo, image}]}` |
| `POST /v1/sessions` | `{avatar?, voice?}` | `201 {session_id, avatar, voice, frame_size, ice_servers}`; `429` if `AVATAR_MAX_SESSIONS` is reached |
| `POST /v1/sessions/{id}/offer` | `{type: "offer", sdp}` | SDP answer. The offer must receive one audio and one video track (`recvonly`), plus optionally a data channel |
| `POST /v1/sessions/{id}/say` | `{text}` or `{audio_wav_base64}`, `voice?`, `emo?` | `202 {utterance_id, tts_ms, audio_ms, lead_silence_ms}`; `409 busy` if another session is speaking; `409 not_connected` |
| `POST /v1/sessions/{id}/stop` | | barge-in: cancels everything queued or playing → `{cancelled: [ids]}` |
| `POST /v1/sessions/{id}/state` | `{state}` | listening body language (below) → `{state, mode, clip, status}`; `400 invalid_state` |
| `GET /v1/sessions/{id}` | | state + per-utterance timings |
| `DELETE /v1/sessions/{id}` | | close |
| `POST /v1/warmup` | `{avatar?, voice?}` (body optional) | loads the renderer if unloaded, renders the avatar once (all avatars if none is warm yet) and probes the TTS voice (skipped if probed < 60 s ago) → `{ok, status, avatar, voice, warmed_avatars, renderer_busy, tts_warmed, model}`; `404 unknown_avatar`. A TTS failure only gives `tts_warmed: false` |
| `POST /v1/cooldown` | `{force?}` (body optional) | frees the GPU now (on-demand mode) → `{ok, unloaded, sessions, model, reason?}`. With open sessions it does nothing (`reason: active_sessions`) unless `force: true`, which closes them first. With `AVATAR_MODEL_IDLE_UNLOAD_S=0`: `reason: persistent_mode` |

**Sentences** posted one after another are queued and play back to back. All sentences of a turn go through one
continuous Ditto stream, so only the first pays the renderer's start-up (about 430 ms to the first frame on an
RTX 3090; the next turn after switching to a *different* avatar adds 150–600 ms). `emo` applies per turn: it is
read when a turn starts.

**Data channel events.** If the offer creates a data channel (any label), the service sends JSON events:

- `{"type": "utterance_start", "id", "text", "start_ms", "lead_silence_ms", "audio_ms"}` when playback reaches the utterance (its first frame and first audio sample);
- `{"type": "utterance_end", "id", "cancelled"}` when it has finished (or was stopped).

- `{"type": "state_applied", "state", "clip", "start_ms"?, "skipped"?}` when a `/state` takes visible effect (below).
- `{"type": "model_loading"}` when a turn has to load the renderer first (only with `AVATAR_MODEL_IDLE_UNLOAD_S`): its
  speech starts about 5 s later than usual.

Use these for captions and turn-taking.

### Listening gestures

`POST /v1/sessions/{id}/state` with `{"state": "<name>"}` tells the avatar what the conversation is doing, so it can
show some listening body language between utterances. The gestures are short clips pre-rendered per avatar at
startup, so they cost no GPU time during a session. They never touch speech.

| `state` | effect |
|---|---|
| `listening` | idle, plus an occasional `nod` or `glance_down` every 6–15 s (persistent) |
| `backchannel` | one `nod` ("mm-hm") |
| `probe` | one head `tilt` |
| `hint`, `wrap_up` | one small closed-mouth `smile` (a `nod` if `smile` is disabled) |
| `silent_thinking` | idle, plus an occasional `glance_down` (at notes) every 10–20 s (persistent) |
| `paused` | idle only, no gestures; drops a pending one (persistent) |

A session starts with no random gestures until the first persistent state. One-shot states don't change the persistent
one. Clips: `nod` 0.8 s, `glance_down` 1.6 s, `tilt` 1.4 s, `smile` 2.0 s.

- **Response** `status`: `pending` (the clip starts at the next idle frame that matches its first frame, usually within
  0.5 s, at most ~2 s), `queued` (an utterance is queued or playing: the clip plays after it, or is dropped if that is
  more than 8 s after the request), `applied` (persistent state set), `unavailable` (gestures disabled).
- **Speech wins.** A clip never starts while an utterance is queued or playing. If `/say` arrives during a clip, the
  utterance starts on time and the clip is cut, just as idle is today. A newer one-shot replaces a pending one.
- **Seamless.** Each clip is a re-render of a stretch of the idle loop itself, with the gesture added on top (head
  pitch / roll / yaw offsets, eye and smile keypoint offsets). It starts on an idle frame (4-frame cross-fade from the
  current one) and hands back to the idle loop on exactly the frame it ends on.
- `state_applied` (data channel) is sent when the clip actually starts (`start_ms` on the media clock), when a
  persistent state is set, or with `skipped: "stale"` when a queued clip was dropped.

**Sync.** Audio for an utterance is released once 8 frames are pre-rendered (`start_ms` on the shared media clock).
The video track then always shows the frame that belongs to the audio position: late frames are dropped and missing
ones hold the previous frame, so lips never drift behind the voice. Measured skew in Chrome was −34…+60 ms. TTS sentences usually start with ~300 ms of
silence. For the first sentence of a turn all but 120 ms of it is skipped (audio and the matching closed-mouth
frames), so the first word comes sooner. Queued sentences keep it as the pause between sentences, and since a stall
while the mouth is closed is invisible, they start right after their first rendered frame.

### Browser notes

- Chrome only feeds a remote WebRTC audio track into Web Audio (`createMediaStreamSource`, e.g. as an echo-canceller
  reference) while the track is also attached to a media element. Attach it to a muted `<audio>` element.
- `network_mode: host` (or a TURN server, via `AVATAR_ICE_SERVERS`) is needed for media to reach browsers. Bridge
  networking advertises container-internal ICE candidates.

## Freeing the GPU when idle

With `AVATAR_MODEL_IDLE_UNLOAD_S=300`, the renderer runs in a child process that is stopped after 300 s without
speech. That frees all of its ~3 GB of VRAM; the main process never touches CUDA. Sessions stay connected and keep
playing the idle loop and gestures, which are pre-rendered and need no GPU.

The renderer is started again:

- by `POST /v1/warmup`, which waits until it is loaded and warm (call it before an interview starts);
- when a session is created or calls `/state` (speech is probably coming soon): ~4 s in the background, usually
  done before the first `/say`;
- otherwise by the turn that needs it. That turn waits ~5 s (measured on an RTX 3090) and the client gets a
  `model_loading` event first.

On an RTX 3090 the child process costs no measurable latency or frame rate (first frame 431–447 ms in the child
vs 433–495 ms in-process, 25 fps both). A crashed renderer is replaced by the next turn. `/health` shows `model`:
`state`, `pid`, `idle_s` and `loads`. With `0` (the default) the renderer stays loaded in the main process, as before.

`POST /v1/cooldown` stops the renderer right away instead of waiting for the timeout (e.g. when an interview ends).
Creating a session also probes its TTS voice in the background (at most once a minute per voice), so a TTS server that
evicted its model while idle (live-stt unloads Kokoro after 10 min) is loading again before the first `/say`.

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
| `AVATAR_MODEL_IDLE_UNLOAD_S` | `0` | free the renderer's VRAM after this many seconds without speech; reloads on demand (~5 s). `0` = stay loaded |
| `AVATAR_IDLE_LOOP_SECONDS` | `8` | length of the pre-rendered idle loop (played ping-pong) |
| `AVATAR_STILL_MOUTH_IDLE` | `1` | hold the lips closed in the idle loop and gesture clips (`0` = Ditto's own idle lip motion) |
| `AVATAR_GESTURES` | `1` | pre-render listening gestures (`0` = idle loop only; `/state` then answers `unavailable` for clips) |
| `AVATAR_GESTURE_CLIPS` | `nod,glance_down,tilt,smile` | which clips to render (drop `smile` to use a nod for `hint`/`wrap_up`) |
| `AVATAR_GESTURE_SCALE` | `1.0` | amplitude of all gestures |
| `AVATAR_GESTURE_ANCHORS` | `2` | idle frames each clip is rendered from: more = shorter wait and closer splices; each adds ~145 frames (~5–6 s on an RTX 3090, estimated) to the first start per avatar |
| `AVATAR_LISTEN_GESTURE_MIN_S` / `_MAX_S` | `6` / `15` | interval of random gestures in `listening` |
| `AVATAR_SAMPLING_STEPS` | `10` | Ditto motion-diffusion steps (5 is a little faster, with more startup jitter) |
| `AVATAR_PREROLL_FRAMES` | `8` | frames rendered before an utterance's audio is released (stall-free start) |
| `AVATAR_TRIM_LEAD_SILENCE` | `1` | skip all but 120 ms of the leading TTS silence of the first sentence of a turn (sooner first word) |
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
- `ditto_engine.py`: the Ditto wrapper (still mouth, re-rendering recorded idle motion for gesture clips);
- `gestures.py`: gesture clip definitions and the per-session idle director (pure Python, unit-tested);
- `engines.py`: the native TensorRT builds;
- `worker.py`: the renderer in a child process, started on demand and stopped when idle;
- `service.py` and `server.py`: sessions and the API.

## Licenses

- **This repository:** Apache-2.0.
- **Ditto:** code and weights are Apache-2.0 (antgroup/ditto-talkinghead). Its face-detection and landmark models
  appear to be derived from InsightFace, whose pretrained models are licensed for non-commercial research only. They
  run only when an avatar is registered. Check this before commercial use.
- **Portraits:** only use photos you have the right to animate. AI-generated faces are the safest choice.
