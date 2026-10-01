"""Bounded inference worker pool with earliest-deadline-first scheduling.

Each job carries the wall-clock time by which its audio must exist to avoid a playback gap:
a stream's first chunk is due now, later chunks are due when the audio already sent runs out.
So under load, new calls' first chunks jump ahead of chunks that still have seconds of slack.
"""
import asyncio
import itertools
import queue
import threading
from collections.abc import Callable


class Scheduler:
    def __init__(self, workers: int) -> None:
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()  # FIFO among equal deadlines; never compare callables
        self.workers = workers
        self.busy = 0
        for i in range(workers):
            threading.Thread(target=self._work, name=f"tts-worker-{i}", daemon=True).start()

    @property
    def depth(self) -> int:
        return self._q.qsize()

    async def run(self, deadline: float, fn: Callable, *args):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._q.put((deadline, next(self._seq), fn, args, fut, loop))
        return await fut  # cancelling the awaiting task cancels fut; the worker then skips the job

    def _work(self) -> None:
        while True:
            _, _, fn, args, fut, loop = self._q.get()
            if fut.cancelled():
                continue
            self.busy += 1
            try:
                result, exc = fn(*args), None
            except Exception as e:  # noqa: BLE001 - delivered to the awaiting coroutine
                result, exc = None, e
            finally:
                self.busy -= 1
            try:
                loop.call_soon_threadsafe(_resolve, fut, result, exc)
            except RuntimeError:  # that loop closed while the job ran (shutdown, a test client): nobody awaits it,
                pass  # and an uncaught raise here would kill this worker for good



def _resolve(fut: asyncio.Future, result, exc) -> None:
    if fut.cancelled():
        return
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(result)
