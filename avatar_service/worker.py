"""The renderer in a child process, so its VRAM can be given back after an idle timeout (AVATAR_MODEL_IDLE_UNLOAD_S).

Ending the process is the only reliable way to free all of it: deleting Ditto in-process leaves the CUDA context
(~350 MiB), and TensorRT 8.6 then segfaults loading the grid-sample-plugin engine (warp_network) a second time.

`RendererProcess` is one child; `OnDemandRenderer` starts one when a render needs it and stops it once nothing has
rendered for `idle_s`. Both have the renderer interface the service uses (`stream`, `render_idle`, `warm`).

Protocol: one duplex Pipe; each side serialises its own sends.
  parent -> child   ("stream", avatar, emo, linger_s)  then  ("item", idx, a16) ... ("cancel",)  ("close", ok)
                    ("idle", avatar, seconds, still_mouth, clips, anchors)   ("warm", avatar)   ("exit",)
  child -> parent   ("status", state) ... ("ready", gpu, engines_dir)  or  ("error", message)   while loading
                    ("segment", idx, first, n)   ("frame", g, shape, dtype) + raw bytes   ("close?", n_received)
                    ("idle_meta", n_frames, {clip: [n per anchor]}, anchors, cost) + frames
                    ("done",)  or  ("error", message)   ending each command
A turn's end is decided in the parent, whose TurnFeed `speak()` appends to: the child asks "close?" with how many
items it has received, and the parent closes the feed only if none is still in flight (TurnFeed.close_if_drained).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from .ditto_engine import TurnFeed
from .gestures import IdleAssets

log = logging.getLogger("avatar.worker")


class RendererDied(RuntimeError):
    """The child process exited (crash, OOM kill, ...)."""


class RemoteError(RuntimeError):
    """A command failed inside the child; the child itself is still fine."""


# ---------------------------------------------------------------------------------------------------------------
# child
# ---------------------------------------------------------------------------------------------------------------
class _RemoteFeed(TurnFeed):
    """The child's view of the parent's TurnFeed: items arrive over the pipe, closing needs the parent's consent."""

    def __init__(self, send: Callable):
        super().__init__()
        self.send = send
        self.received = 0
        self.replies: queue.Queue = queue.Queue()

    def deliver(self, msg: tuple) -> None:
        if msg[0] == "item":
            with self.cond:
                self.received += 1
                self.put((msg[1], msg[2]))
        elif msg[0] == "cancel":
            self.cancel()
            self.replies.put(True)      # unblocks a pending close_if_empty
        elif msg[0] == "close":
            self.replies.put(msg[1])

    def close_if_empty(self) -> bool:
        with self.cond:
            if self.cancelled:
                self.closed = True
                return True
            if self.items:
                return False
            n = self.received
        self.send(("close?", n))
        if not self.replies.get():
            return False                # an item is on its way; the stream waits for it
        with self.cond:
            self.closed = True
        return True


def _load(cfg: dict, status: Callable[[str], None]):
    if cfg["fake"]:
        from .ditto_engine import FakeRenderer
        return FakeRenderer(), "fake", None
    from .ditto_engine import DittoRenderer
    from .engines import ensure_engines
    status("building_engines")
    engines = ensure_engines(cfg["ditto_models"], cfg["data_dir"], native=cfg["native"])
    status("loading")
    import torch
    t = time.time()
    r = DittoRenderer(cfg["ditto_dir"], cfg["ditto_models"], engines, cfg["sampling_steps"])
    log.info("renderer process %d loaded in %.1f s", os.getpid(), time.time() - t)
    return r, torch.cuda.get_device_name(0), engines


def _child_main(conn, cfg: dict) -> None:
    logging.basicConfig(level=cfg.get("log_level", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    lock = threading.Lock()

    def send(msg, data=None):
        with lock:
            conn.send(msg)
            if data is not None:
                conn.send_bytes(data)

    def send_frame(g, img):
        img = np.ascontiguousarray(img)
        # a flat byte view: send_bytes() frames the message with len(), which for an ndarray is its first dimension
        send(("frame", g, img.shape, img.dtype.str), memoryview(img.reshape(-1)).cast("B"))

    try:
        renderer, gpu, engines = _load(cfg, lambda s: send(("status", s)))
    except Exception as e:  # noqa: BLE001 - reported to the parent
        log.exception("renderer failed to load")
        send(("error", f"{type(e).__name__}: {e}"))
        return
    send(("ready", gpu, engines))

    commands: queue.Queue = queue.Queue()
    current: list = [None]                      # _RemoteFeed of the running (or last) stream

    def reader():
        try:
            while True:
                msg = conn.recv()
                if msg[0] in ("item", "cancel", "close"):
                    if current[0] is not None:
                        current[0].deliver(msg)
                    continue
                if msg[0] == "stream":          # set up here, so items sent right after can't miss it
                    current[0] = _RemoteFeed(send)
                    msg = (*msg, current[0])
                commands.put(msg)
        except (EOFError, OSError):             # parent gone
            commands.put(("exit",))

    threading.Thread(target=reader, name="pipe-reader", daemon=True).start()
    while True:
        msg = commands.get()
        op = msg[0]
        if op == "exit":
            break
        try:
            if op == "stream":
                _, avatar, emo, linger_s, feed = msg
                renderer.stream(avatar, feed, lambda key, first, n: send(("segment", key, first, n)), send_frame,
                                emo=emo, linger_s=linger_s)
            elif op == "warm":
                renderer.warm(msg[1])
            elif op == "idle":
                _, avatar, seconds, still_mouth, clips, anchors = msg
                a = renderer.render_idle(avatar, seconds, still_mouth=still_mouth, clips=clips, anchors=anchors)
                send(("idle_meta", len(a.frames), {k: [len(v) for v in vs] for k, vs in a.clips.items()},
                      list(a.anchors), a.cost))
                for f in a.frames:
                    send_frame(-1, f)
                for vs in a.clips.values():
                    for v in vs:
                        for f in v:
                            send_frame(-1, f)
            else:
                raise ValueError(f"unknown command {op!r}")
            send(("done",))
        except Exception as e:  # noqa: BLE001 - reported to the parent
            log.exception("renderer command %s failed", op)
            send(("error", f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------------------------------------------
class RendererProcess:
    """One renderer child process. Not thread-safe: one command at a time (OnDemandRenderer serialises)."""

    def __init__(self, cfg: dict, on_status: Callable[[str], None] = lambda s: None):
        ctx = mp.get_context("spawn")   # never fork: the parent runs threads and an event loop
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_child_main, args=(child, cfg), name="avatar-renderer", daemon=True)
        self.proc.start()
        child.close()                   # so recv() sees EOF when the child dies
        self.send_lock = threading.Lock()
        try:
            while True:                 # no timeout: a first start builds TensorRT engines for minutes
                msg = self._recv()
                if msg[0] == "status":
                    on_status(msg[1])
                elif msg[0] == "ready":
                    _, self.gpu, self.engines = msg
                    break
                else:
                    raise RemoteError(msg[1] if msg[0] == "error" else f"unexpected {msg!r}")
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.is_alive()

    def _send(self, msg) -> None:
        try:
            with self.send_lock:
                self.conn.send(msg)
        except (OSError, ValueError) as e:
            raise RendererDied(f"renderer process exited ({self.proc.exitcode})") from e

    def _recv(self):
        try:
            while not self.conn.poll(1.0):
                if not self.proc.is_alive():
                    raise EOFError
            return self.conn.recv()
        except (EOFError, OSError) as e:
            self.proc.join(1)
            raise RendererDied(f"renderer process exited ({self.proc.exitcode})") from e

    def _recv_frame(self, msg) -> np.ndarray:
        _, _, shape, dtype = msg
        buf = bytearray(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        try:
            self.conn.recv_bytes_into(buf)
        except (EOFError, OSError) as e:
            raise RendererDied("renderer process exited") from e
        return np.frombuffer(buf, dtype).reshape(shape)    # writable (backed by the bytearray)

    def _wait_done(self, handle: Callable = lambda msg: None) -> None:
        while True:
            msg = self._recv()
            if msg[0] == "done":
                return
            if msg[0] == "error":
                raise RemoteError(msg[1])
            handle(msg)

    def warm(self, avatar) -> None:
        self._send(("warm", avatar))
        self._wait_done()

    def render_idle(self, avatar, seconds, still_mouth=True, clips=None, anchors=2) -> IdleAssets:
        self._send(("idle", avatar, seconds, still_mouth, clips, anchors))
        frames: list = []
        meta: list = []

        def handle(msg):
            if msg[0] == "idle_meta":
                meta.append(msg)
            elif msg[0] == "frame":
                frames.append(self._recv_frame(msg))

        self._wait_done(handle)
        _, n, lens, anchor_list, cost = meta[0]
        out = IdleAssets(frames[:n], anchors=anchor_list, cost=cost)
        i = n
        for name, ns in lens.items():
            out.clips[name] = []
            for k in ns:
                out.clips[name].append(frames[i:i + k])
                i += k
        return out

    def stream(self, avatar, feed: TurnFeed, on_segment, on_frame, emo=None, linger_s: float = 1.0) -> None:
        """Same contract as DittoRenderer.stream; `feed` stays in this process and is forwarded item by item."""
        keys: list = []                 # idx -> the caller's segment key (never pickled)
        done = threading.Event()
        self._send(("stream", avatar, emo, linger_s))

        def pump():
            try:
                while not done.is_set():
                    item = feed.next(0.2)
                    if item is not None:
                        keys.append(item[0])
                        self._send(("item", len(keys) - 1, item[1]))
                    elif feed.cancelled:
                        self._send(("cancel",))
                        return
            except RendererDied:
                pass

        t = threading.Thread(target=pump, name="feed-pump", daemon=True)
        t.start()
        try:
            while True:
                msg = self._recv()
                op = msg[0]
                if op == "frame":
                    on_frame(msg[1], self._recv_frame(msg))
                elif op == "segment":
                    on_segment(keys[msg[1]], msg[2], msg[3])
                elif op == "close?":
                    self._send(("close", feed.close_if_drained(msg[1])))
                elif op == "done":
                    return
                elif op == "error":
                    raise RemoteError(msg[1])
                else:
                    raise RuntimeError(f"unexpected message from renderer: {op!r}")
        finally:
            done.set()
            t.join()

    def close(self, timeout: float = 5.0) -> None:
        if self.proc.is_alive():
            try:
                self._send(("exit",))
            except RendererDied:
                pass
            self.proc.join(timeout)
            if self.proc.is_alive():
                self.proc.kill()
        self.proc.join(timeout)
        self.conn.close()


class OnDemandRenderer:
    """Starts the renderer process when something needs it; stops it after `idle_s` without a render."""

    def __init__(self, cfg: dict, idle_s: float, on_state: Callable[[str], None] = lambda s: None):
        self.cfg, self.idle_s, self.on_state = cfg, idle_s, on_state
        self.proc: Optional[RendererProcess] = None
        self.lock = threading.Lock()        # held for loading, unloading and every render call
        self.state = "unloaded"             # unloaded | loading | loaded (| building_engines while loading)
        self.error: Optional[str] = None
        self.gpu: Optional[str] = None
        self.engines: Optional[str] = None
        self.last_used = time.time()
        self.loads = 0
        self._preloading = False
        self._closed = threading.Event()
        threading.Thread(target=self._reaper, name="renderer-reaper", daemon=True).start()

    @property
    def loaded(self) -> bool:
        return self.proc is not None and self.proc.alive()

    def _set_state(self, state: str) -> None:
        self.state = state
        self.on_state(state)

    def _ensure(self) -> RendererProcess:   # with self.lock
        if self.loaded:
            return self.proc
        if self.proc is not None:           # died since the last call
            log.warning("renderer process %s exited (%s); starting a new one", self.proc.pid, self.proc.proc.exitcode)
            self.proc.close()
            self.proc = None
        self._set_state("loading")
        t = time.time()
        try:
            self.proc = RendererProcess(self.cfg, on_status=self._set_state)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self._set_state("unloaded")
            raise
        self.gpu, self.engines, self.error = self.proc.gpu, self.proc.engines, None
        self.loads += 1
        log.info("renderer loaded in %.1f s (pid %s)", time.time() - t, self.proc.pid)
        self._set_state("loaded")
        return self.proc

    def _call(self, fn):
        with self.lock:
            proc = self._ensure()
            try:
                return fn(proc)
            except RemoteError:
                raise                       # the child finished the command cleanly
            except BaseException:
                # died, or the pipe is mid-command (e.g. a callback raised): a fresh process next time
                proc.close(timeout=2)
                self.proc = None
                self._set_state("unloaded")
                raise
            finally:
                self.last_used = time.time()

    def load(self) -> None:
        self._call(lambda p: None)

    def preload(self, warm_avatar=None) -> None:
        """Start loading in the background if unloaded (e.g. a client connected: speech is likely soon)."""
        if self.loaded or self._preloading or self._closed.is_set():
            return
        self._preloading = True

        def run():
            try:
                self._call(lambda p: p.warm(warm_avatar) if warm_avatar is not None else None)
            except Exception:  # noqa: BLE001 - the next render retries and reports
                log.exception("renderer preload failed")
            finally:
                self._preloading = False

        threading.Thread(target=run, name="renderer-preload", daemon=True).start()

    def warm(self, avatar) -> None:
        self._call(lambda p: p.warm(avatar))

    def render_idle(self, avatar, seconds, still_mouth=True, clips=None, anchors=2) -> IdleAssets:
        return self._call(lambda p: p.render_idle(avatar, seconds, still_mouth=still_mouth, clips=clips, anchors=anchors))

    def stream(self, avatar, feed, on_segment, on_frame, emo=None, linger_s: float = 1.0) -> None:
        self._call(lambda p: p.stream(avatar, feed, on_segment, on_frame, emo=emo, linger_s=linger_s))

    def unload(self, if_idle: bool = False) -> bool:
        """Stop the process. With `if_idle`: only if nothing is rendering or loading and it has been idle `idle_s`."""
        if not self.lock.acquire(blocking=not if_idle):
            return False
        try:
            idle = time.time() - self.last_used
            if self.proc is None or (if_idle and idle < self.idle_s):
                return False
            pid = self.proc.pid
            self.proc.close()
            self.proc = None
            self._set_state("unloaded")
        finally:
            self.lock.release()
        log.info("renderer process %s stopped after %.0f s idle: VRAM freed", pid, idle)
        return True

    def _reaper(self) -> None:
        while not self._closed.wait(min(5.0, max(0.2, self.idle_s / 4))):
            if self.proc is not None and time.time() - self.last_used >= self.idle_s:
                self.unload(if_idle=True)

    def close(self) -> None:
        self._closed.set()
        self.unload()
