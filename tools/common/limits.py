"""Run a computation in a child process under a memory limit and a time limit.

A tool that needs more memory than the machine has must be reported as out of
memory without the machine getting there. The computation therefore runs in its
own process, which watches its memory footprint from a thread and stops itself
as soon as the footprint exceeds the limit; the parent enforces the time limit.
(The footprint is read inside the process because macOS does not let a parent
read it, and the resident set size seen from outside leaves out compressed
memory.)
"""

import multiprocessing
import os
import threading
import time
import traceback

import psutil

GB = 1e9
_MEMORY_EXIT = 86  # exit code of a child that stopped itself at the memory limit


class MemoryLimitExceeded(Exception):
    """The computation used more memory than allowed, or an allocation was refused."""

    def __init__(self, used, limit, elapsed, note=""):
        self.used, self.limit, self.elapsed, self.note = used, limit, elapsed, note
        super().__init__(f"memory use {used / GB:.3g} GB above the limit of {limit / GB:.3g} GB after {elapsed:.3g} s")


class TimeLimitExceeded(Exception):
    def __init__(self, timeout, peak):
        self.timeout, self.peak = timeout, peak
        super().__init__(f"no result within {timeout:g} s")


class ChildFailed(RuntimeError):
    """The computation raised an exception or its process died."""


def footprint(process=None):
    """Memory footprint of a process in bytes (of this process by default)."""
    process = process or psutil.Process()
    try:
        return process.memory_full_info().uss  # includes compressed memory on macOS
    except (psutil.AccessDenied, AttributeError):
        return process.memory_info().rss


def is_allocation_failure(exc):
    """Whether an exception says that memory could not be allocated (numpy MemoryError, XLA RESOURCE_EXHAUSTED)."""
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or "resource_exhausted" in text


def _child(conn, func, args, limit, poll_s):
    peak = [0]
    lock = threading.Lock()  # one message through the pipe

    def watch():
        me = psutil.Process()
        while True:
            used = footprint(me)
            peak[0] = max(peak[0], used)
            if limit is not None and used > limit:
                with lock:
                    conn.send(("memory", "", used))
                    os._exit(_MEMORY_EXIT)  # no clean-up: the memory has to go now
            time.sleep(poll_s)

    threading.Thread(target=watch, daemon=True).start()
    try:
        message = ("ok", func(*args))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        kind = "memory" if is_allocation_failure(exc) else "error"
        message = (kind, traceback.format_exc() if kind == "error" else f"{type(exc).__name__}: {exc}")
    with lock:
        conn.send((*message, peak[0]))
        conn.close()


def run_limited(func, args=(), memory_limit=None, timeout=None, poll_s=0.02):
    """func(*args) in a new process; returns (result, peak memory footprint in bytes).

    func must be defined at module level (the process is spawned, not forked) and its result must
    be picklable. memory_limit is in bytes, timeout in seconds. Raises MemoryLimitExceeded,
    TimeLimitExceeded or ChildFailed.
    """
    ctx = multiprocessing.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(send, func, args, memory_limit, poll_s), daemon=True)
    t0 = time.perf_counter()
    proc.start()
    send.close()
    seen = 0  # largest resident set size seen from here: a lower bound of the footprint, kept as a second guard
    try:
        watched = psutil.Process(proc.pid)
        while True:
            if recv.poll(0.05):
                try:
                    kind, payload, peak = recv.recv()
                except EOFError:  # the process died without a message
                    proc.join()
                    raise ChildFailed(f"the process ended with exit code {proc.exitcode}") from None
                elapsed = time.perf_counter() - t0
                if kind == "ok":
                    return payload, peak
                if kind == "memory":
                    raise MemoryLimitExceeded(peak, memory_limit if memory_limit is not None else peak, elapsed, payload)
                raise ChildFailed(payload)
            elapsed = time.perf_counter() - t0
            try:
                seen = max(seen, watched.memory_info().rss)
            except psutil.Error:
                pass  # the process is ending; its message, if any, is read above
            if memory_limit is not None and seen > memory_limit:
                raise MemoryLimitExceeded(seen, memory_limit, elapsed)
            if timeout is not None and elapsed > timeout:
                raise TimeLimitExceeded(timeout, seen)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        recv.close()
