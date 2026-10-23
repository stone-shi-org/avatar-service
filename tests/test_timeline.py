import numpy as np

from avatar_service.audio import SAMPLES_PER_FRAME
from avatar_service.session import Timeline, Utterance

IDLE = np.zeros((2, 2, 3), np.uint8)


def frame(i):
    return np.full((2, 2, 3), i % 250 + 1, np.uint8)


def utt(frames=10):
    return Utterance(text="hi", audio48=np.ones(frames * SAMPLES_PER_FRAME, np.int16))


def feed(tl, u, n):
    for _ in range(n):
        u.frames.append((u.rendered, frame(u.rendered)))
        u.rendered += 1
        tl.frame_ready(u)


def test_release_waits_for_preroll():
    events = []
    tl = Timeline(preroll=3, on_event=events.append)
    u = utt()
    tl.add(u)
    feed(tl, u, 2)
    assert u.start_tick is None and not events
    feed(tl, u, 1)
    assert u.start_tick == 1 and not events
    tl.next_video(lambda t: IDLE)       # tick 0: still idle
    assert not events
    tl.next_video(lambda t: IDLE)       # tick 1: playback reaches the utterance
    assert events[0]["type"] == "utterance_start" and events[0]["start_ms"] == 40
    # audio for the utterance starts exactly at its start tick
    assert sum(len(a) for a in tl.audio_q) == (u.start_tick * SAMPLES_PER_FRAME) + len(u.audio48)


def test_video_follows_audio_clock_dropping_late_and_holding_missing():
    tl = Timeline(preroll=2)
    u = utt(10)
    tl.add(u)
    feed(tl, u, 2)                      # released at tick 1
    assert tl.next_video(lambda t: IDLE) is IDLE        # tick 0: before start -> idle
    f = tl.next_video(lambda t: IDLE)                   # tick 1 -> frame 0
    assert f[0, 0, 0] == frame(0)[0, 0, 0]
    tl.next_video(lambda t: IDLE)                       # tick 2 -> frame 1
    held = tl.next_video(lambda t: IDLE)                # tick 3 -> frame 2 missing: hold frame 1
    assert held[0, 0, 0] == frame(1)[0, 0, 0]
    feed(tl, u, 4)                                      # frames 2..5 arrive late
    f = tl.next_video(lambda t: IDLE)                   # tick 4 -> frame 3 (frame 2 dropped)
    assert f[0, 0, 0] == frame(3)[0, 0, 0]


def test_finish_ends_with_the_audio_and_drops_surplus_frames():
    events = []
    tl = Timeline(preroll=1, on_event=events.append)
    u = utt(5)
    tl.add(u)
    feed(tl, u, 7)                      # Ditto produced 2 more (silent) frames than the audio covers
    tl.finish(u, 7)
    assert u.end_tick == u.start_tick + 5
    assert tl.audio_queued_end == u.end_tick * SAMPLES_PER_FRAME
    for _ in range(u.end_tick + 1):
        tl.next_video(lambda t: IDLE)
    assert not tl.busy and events[-1] == {"type": "utterance_end", "id": u.id, "cancelled": False}


def test_back_to_back_utterances_do_not_overlap():
    tl = Timeline(preroll=1)
    a, b = utt(10), utt(4)
    tl.add(a)
    tl.add(b)
    feed(tl, a, 10)
    tl.finish(a, 10)
    feed(tl, b, 1)
    assert b.start_tick == a.end_tick


def test_stop_cancels_everything():
    events = []
    tl = Timeline(preroll=1, on_event=events.append)
    u = utt()
    tl.add(u)
    feed(tl, u, 3)
    assert tl.stop() == [u.id]
    assert u.cancelled and not tl.busy and not tl.audio_q
    assert events[-1]["cancelled"] is True
    out = tl.next_audio(960)
    assert not out.any()


def test_audio_pull_is_exact_length():
    tl = Timeline()
    tl.audio_q.append(np.ones(1000, np.int16))
    assert len(tl.next_audio(960)) == 960
    rest = tl.next_audio(960)
    assert rest[:40].all() and not rest[40:].any()


def test_finish_pads_partial_last_frame_of_audio():
    tl = Timeline(preroll=1)
    u = Utterance(text="hi", audio48=np.ones(4 * SAMPLES_PER_FRAME + 100, np.int16))  # 4.05 frames of audio
    tl.add(u)
    feed(tl, u, 3)                      # fewer frames than the audio needs: video holds the last one
    tl.finish(u, 3)
    assert u.end_tick == u.start_tick + 5
    assert tl.audio_queued_end == u.end_tick * SAMPLES_PER_FRAME


def test_leading_silence_is_trimmed_and_frames_stay_in_sync():
    events = []
    tl = Timeline(preroll=2, on_event=events.append)
    u = utt(20)
    u.lead_silence_ms = 300                       # 7 whole frames of silence -> skip 7 - 3 = 4 frames
    tl.add(u)
    feed(tl, u, 10)                               # released at tick 0: playback starts at tick 1
    assert u.trim_frames == 4 and u.start_tick == 1 - 4
    # one frame of alignment silence, then the audio from frame 4 on
    assert sum(len(a) for a in tl.audio_q) == SAMPLES_PER_FRAME + (20 - 4) * SAMPLES_PER_FRAME
    assert tl.next_video(lambda t: IDLE) is IDLE and not events     # tick 0: idle, trimmed frames never shown
    f = tl.next_video(lambda t: IDLE)                               # tick 1: frame 4, with frame 4's audio
    assert f[0, 0, 0] == frame(4)[0, 0, 0]
    assert events[0]["type"] == "utterance_start" and events[0]["start_ms"] == 40
    tl.finish(u, 20)
    assert u.end_tick == u.start_tick + 20 and tl.audio_queued_end == u.end_tick * SAMPLES_PER_FRAME


def test_trim_can_be_disabled():
    tl = Timeline(preroll=1, trim_lead=False)
    u = utt(10)
    u.lead_silence_ms = 300
    tl.add(u)
    feed(tl, u, 1)
    assert u.trim_frames == 0 and u.start_tick == 1


def test_queued_sentence_keeps_its_pause_and_starts_after_one_frame():
    tl = Timeline(preroll=8)
    a, b = utt(10), utt(20)
    b.lead_silence_ms = 300                       # 7 frames of lead-in silence
    tl.add(a)
    tl.add(b)
    feed(tl, a, 10)
    tl.finish(a, 10)
    feed(tl, b, 1)                                # one frame is enough: the first 7 are silent anyway
    assert b.start_tick == a.end_tick and b.trim_frames == 0


def test_turn_feed_never_loses_a_sentence():
    from avatar_service.ditto_engine import TurnFeed
    f = TurnFeed()
    assert f.put("a") and f.next(0) == "a"
    assert f.next(0.01) is None
    assert f.close_if_empty()            # stream ends...
    assert not f.put("b")                # ...so the caller must start a new stream
    g = TurnFeed()
    g.put("x")
    assert not g.close_if_empty()        # a sentence arrived in time: stream must go on
    assert g.next(0) == "x"
    g.put("y")
    assert g.cancel() == ["y"] and g.next(0) is None and not g.put("z")
