import random

import numpy as np
import pytest

from avatar_service import gestures as G
from avatar_service.gestures import Director, IdleAssets


def img(v):
    return np.full((4, 4, 3), v, np.uint8)


def assets(n=20, clip_len=5, anchors=(3,), cost=None, names=("nod", "glance_down", "tilt", "smile")):
    """Idle frame i has value i; clip frames of anchor a: value 100 + 10*a_index + j."""
    frames = [img(i) for i in range(n)]
    clips = {c: [[img(100 + 10 * ai + j) for j in range(clip_len)] for ai in range(len(anchors))] for c in names}
    if cost is None:
        cost = np.full((len(anchors), n), 50.0, np.float32)
        for ai, a in enumerate(anchors):
            cost[ai, a] = 0.0
    return IdleAssets(frames, clips, list(anchors), cost)


def play(d, ticks, start=0):
    return [int(d.frame(t)[0, 0, 0]) for t in range(start, start + ticks)]


# ---- clip specs ----

@pytest.mark.parametrize("name", G.CLIP_NAMES)
def test_clip_controls_start_and_end_at_zero(name):
    ctrls = G.clip_controls(name, scale=np.float64(1.0))
    for c in ctrls:      # Ditto's engines are float32: nothing float64 may reach them
        for k, v in c.items():
            assert type(v) is float or v.dtype == np.float32, (k, type(v))
    assert 15 <= len(ctrls) <= 60
    for c in (ctrls[0], ctrls[-1]):
        for k, v in c.items():
            assert np.allclose(v, 0), (name, k)
    peak = ctrls[len(ctrls) // 2]
    assert any(np.abs(v).max() > 0 for v in peak.values())


def test_envelopes():
    b = G.bump(21)
    assert b[0] == pytest.approx(0) and b[-1] == pytest.approx(0) and b[10] == pytest.approx(1)
    p = G.plateau(30, 8)
    assert p[0] == 0 and p[-1] == pytest.approx(0) and np.all(np.array(p[8:22]) == 1)


def test_exp_offset_touches_expected_keypoints():
    e = G.exp_offset(smile=1).reshape(21, 3)
    assert e[14, 1] == pytest.approx(-0.02) and e[0].sum() == 0
    down = G.exp_offset(pupil_y=-10).reshape(21, 3)
    assert down[11, 1] > 0 and down[15, 1] > 0


def test_pick_anchors_avoid_blinks_and_leave_room():
    n = 100
    poses = np.zeros((n, 5))
    poses[:, 0] = np.sin(np.arange(n) / 8) * 3
    blink = np.zeros(n, bool)
    blink[40:55] = True
    an = G.pick_anchors(poses, blink, 3, clip_len=30)
    assert len(an) == 3 and len(set(an)) == 3
    assert all(a + 30 < n and not blink[a] for a in an)
    cost = G.pose_cost(poses, blink, an)
    assert cost.shape == (3, n) and all(cost[i, a] == 0 for i, a in enumerate(an))
    assert cost[0, 45] >= G.BLINK_COST


# ---- director ----

def test_idle_ping_pong_without_gestures():
    d = Director(IdleAssets([img(i) for i in range(4)]))
    assert play(d, 8) == [0, 1, 2, 3, 2, 1, 0, 1]
    assert d.set_state("backchannel")["status"] == "unavailable"


def test_no_assets_returns_none():
    assert Director(None).frame(0) is None


def test_one_shot_splices_at_anchor_and_resumes_after_clip():
    events = []
    d = Director(assets(anchors=(3,)), on_event=events.append)
    d.XFADE = 0
    r = d.set_state("backchannel")
    assert r == {"state": "backchannel", "mode": "idle", "clip": "nod", "status": "pending"}
    out = play(d, 12)
    # idle 0,1,2 then the clip replaces idle frames 3..7, then idle continues at 8
    assert out == [0, 1, 2, 100, 101, 102, 103, 104, 8, 9, 10, 11]
    assert events == [{"type": "state_applied", "state": "backchannel", "clip": "nod", "start_ms": 120}]


def test_crossfade_blends_into_clip():
    d = Director(assets(anchors=(0,)))
    d.set_state("probe")
    first = d.frame(0)[0, 0, 0]
    assert 0 < first < 100                 # blend of clip frame 0 (100) and idle frame 0
    assert d.status()["playing"] == "tilt"


def test_waits_for_matching_pose_then_relaxes():
    n = 50
    cost = np.full((1, n), 3.0, np.float32)     # nothing matches well
    d = Director(assets(n=n, anchors=(10,), cost=cost))
    d.set_state("backchannel")
    ticks = 0
    while d.status()["pending"]:
        d.frame(ticks)
        ticks += 1
        assert ticks < 100
    # threshold 0.6 + 2.0/s reaches 3.0 after 1.2 s = 30 frames
    assert 29 <= ticks <= 32


def test_clip_never_starts_while_speaking_and_is_queued():
    busy = [True]
    d = Director(assets(anchors=(0,)), busy=lambda: busy[0])
    d.XFADE = 0
    assert d.set_state("hint")["status"] == "queued"
    assert all(v < 100 for v in play(d, 10))
    busy[0] = False
    out = play(d, 60, start=10)
    assert any(v >= 100 for v in out)


def test_stale_queued_one_shot_is_dropped():
    busy = [True]
    events = []
    d = Director(assets(anchors=(0,)), busy=lambda: busy[0], on_event=events.append, queue_s=1.0)
    d.set_state("wrap_up")
    play(d, 40)                     # 1.6 s of speech
    busy[0] = False
    assert all(v < 100 for v in play(d, 60, start=40))
    assert events[-1]["skipped"] == "stale"


def test_gap_in_ticks_drops_half_played_clip():
    d = Director(assets(anchors=(0,), clip_len=10))
    d.XFADE = 0
    d.set_state("backchannel")
    play(d, 3)
    assert d.status()["playing"] == "nod"
    d.frame(50)                     # an utterance was shown for ticks 3..49
    assert d.status()["playing"] is None


def test_hint_falls_back_to_nod_without_smile():
    d = Director(assets(names=("nod",)))
    assert d.set_state("hint")["clip"] == "nod"
    assert d.set_state("probe")["status"] == "unavailable"


def test_listening_mode_adds_occasional_gestures_and_paused_stops_them():
    d = Director(assets(n=40, anchors=(0, 20)), rng=random.Random(3), listen_interval=(6, 15))
    d.XFADE = 0
    assert d.set_state("listening")["status"] == "applied"
    out = play(d, 25 * 60)
    starts = sum(1 for a, b in zip(out, out[1:]) if a < 100 <= b)
    assert 4 <= starts <= 11          # one every 6-15 s over a minute
    d.set_state("paused")
    out = play(d, 25 * 60, start=25 * 60)
    assert sum(1 for a, b in zip(out, out[1:]) if a < 100 <= b) <= 1   # at most one already in flight


def test_unknown_state_raises():
    with pytest.raises(ValueError):
        Director(assets()).set_state("dancing")


# ---- Ditto MotionStitch patch (no GPU: a stand-in base class) ----

class FakeMotionStitch:
    def __init__(self):
        self.d0 = {"exp": np.zeros((1, 63), np.float32)}
        self.overall_ctrl_info = {"delta_pitch": 2}
        self.idx = 0
        self.calls = []

    def __call__(self, x_s_info, x_d_info, **kw):
        self.calls.append((x_d_info, dict(kw), self.idx))
        self.idx += 1
        return None, None


def test_patch_motion_stitch_holds_lips_and_adds_controls():
    from avatar_service.ditto_engine import LIP_DIMS, patch_motion_stitch
    ms = FakeMotionStitch()
    patch_motion_stitch(ms)
    patch_motion_stitch(ms)                      # idempotent
    exp = np.full((1, 63), 0.5, np.float32)
    ms.recorder = []
    ms({}, {"exp": exp}, mouth_alpha=0.0, add_pitch=np.float64(3.0), eye_idx=7, add_exp=np.ones((1, 63)))
    x_d, kw, idx = ms.calls[-1]
    assert type(kw["delta_pitch"]) is float and kw["delta_exp"].dtype == np.float32
    assert np.all(x_d["exp"][0, LIP_DIMS] == 0)            # lips back at d0 (= the portrait)
    others = np.setdiff1d(np.arange(63), LIP_DIMS)
    assert np.all(x_d["exp"][0, others] == 0.5)            # eyes etc. untouched
    assert kw == {"delta_pitch": 5.0, "delta_exp": kw["delta_exp"]} and idx == 7
    assert len(ms.recorder) == 1 and np.all(ms.recorder[0][0]["exp"] == 0.5)   # raw motion recorded
    ms.recorder = None
    ms({}, {"exp": exp})                                   # speech: untouched
    x_d, kw, _ = ms.calls[-1]
    assert x_d["exp"] is exp and kw == {}
