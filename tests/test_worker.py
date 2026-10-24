"""The renderer child process (worker.py), with the fake renderer: protocol, turn closing, cancel, idle unload."""
import threading
import time

import numpy as np
import pytest

from avatar_service.avatars import Avatar
from avatar_service.ditto_engine import FakeRenderer, TurnFeed
from avatar_service.worker import OnDemandRenderer, RemoteError, RendererProcess

AV = Avatar("a", "/nonexistent.png", "A")
CFG = {"fake": True, "log_level": "WARNING"}
HOP = 640


def tone(sec):
    t = np.arange(int(16000 * sec)) / 16000
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


@pytest.fixture(scope="module")
def proc():
    p = RendererProcess(CFG)
    yield p
    p.close()


def collect(renderer, feed, **kw):
    segs, frames = [], []
    renderer.stream(AV, feed, lambda k, first, n: segs.append((k, first, n)),
                    lambda g, img: frames.append((g, img)), **kw)
    return segs, frames


def test_feed_close_if_drained():
    f = TurnFeed()
    f.put(("a", 1))
    assert f.next(0) == ("a", 1) and f.taken == 1
    assert f.close_if_drained(0) is False and not f.closed      # the item is still on its way to the consumer
    assert f.close_if_drained(1) is True and f.put(("b", 2)) is False


def test_stream_segments_frames_and_close(proc):
    feed = TurnFeed()
    feed.put(("u1", tone(0.4)))
    late = threading.Timer(0.3, lambda: feed.put(("u2", tone(0.2))))   # arrives during the linger
    late.start()
    segs, frames = collect(proc, feed, linger_s=0.8)
    late.join()
    assert [s[0] for s in segs] == ["u1", "u2"]
    assert segs[0][1:] == (0, 10) and segs[1][1:] == (10, 5)
    assert [g for g, _ in frames] == list(range(15))
    assert frames[0][1].shape == (360, 640, 3) and frames[0][1].flags.writeable
    assert feed.closed and feed.put(("u3", tone(0.1))) is False    # caller starts a new stream instead


def test_stream_cancel(proc):
    feed = TurnFeed()
    feed.put(("u1", tone(3.0)))
    threading.Timer(0.1, feed.cancel).start()
    t = time.time()
    segs, frames = collect(proc, feed)
    assert time.time() - t < 2.0 and len(frames) < 75
    segs, frames = collect(proc, (lambda f: (f.put(("u", tone(0.2))), f)[1])(TurnFeed()), linger_s=0.1)
    assert len(frames) == 5                                         # the process is still usable


def test_render_idle_matches_in_process(proc):
    clips = {"nod": [{}] * 10, "tilt": [{}] * 8}
    remote = proc.render_idle(AV, 1, clips=clips, anchors=2)
    local = FakeRenderer().render_idle(AV, 1, clips=clips, anchors=2)
    assert len(remote.frames) == len(local.frames) and remote.anchors == local.anchors
    assert {k: [len(v) for v in vs] for k, vs in remote.clips.items()} == \
           {k: [len(v) for v in vs] for k, vs in local.clips.items()}
    assert np.array_equal(remote.clips["tilt"][0][0], local.clips["tilt"][0][0])
    assert np.array_equal(remote.cost, local.cost)


def test_remote_error_keeps_process(proc):
    with pytest.raises(RemoteError):
        proc.warm(None)          # fails inside the child (no avatar)
    proc.warm(AV)


def test_on_demand_unload_reload_and_crash():
    states = []
    r = OnDemandRenderer(CFG, idle_s=0.5, on_state=states.append)
    try:
        assert not r.loaded and r.state == "unloaded"
        r.load()
        assert r.loaded and r.loads == 1
        for _ in range(50):
            if not r.loaded:
                break
            time.sleep(0.1)
        assert not r.loaded and r.state == "unloaded"                # stopped after idle_s
        feed = TurnFeed()
        feed.put(("u", tone(0.2)))
        segs, frames = collect(r, feed, linger_s=0.1)                # loads again on demand
        assert len(frames) == 5 and r.loads == 2
        r.proc.proc.kill()                                           # crash: the next call starts a fresh one
        r.proc.proc.join()
        r.warm(AV)
        assert r.loads == 3 and r.loaded
        assert states[:2] == ["loading", "loaded"] and "unloaded" in states
    finally:
        r.close()
    assert not r.loaded


def test_on_demand_never_unloads_mid_render():
    r = OnDemandRenderer(CFG, idle_s=0.3)
    try:
        r.load()
        feed = TurnFeed()
        feed.put(("u", tone(2.0)))
        segs, frames = collect(r, feed, linger_s=1.2)   # the stream outlasts idle_s several times over
        assert len(frames) == 50 and r.loads == 1
    finally:
        r.close()


def test_preload():
    r = OnDemandRenderer(CFG, idle_s=30)
    try:
        r.preload(AV)
        for _ in range(100):
            if r.loaded and not r._preloading:
                break
            time.sleep(0.05)
        assert r.loaded and r.loads == 1
        r.preload(AV)                # no-op when loaded
        time.sleep(0.1)
        assert r.loads == 1
    finally:
        r.close()
