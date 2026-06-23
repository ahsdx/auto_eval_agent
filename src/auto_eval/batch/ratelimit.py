"""异步限流：每模型一个并发信号量 + RPM 节流。"""
from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, concurrency: int = 4, rpm: int | None = None):
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._min_interval = (60.0 / rpm) if rpm else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0  # 下一个允许发请求的时刻

    async def acquire(self) -> None:
        await self._sem.acquire()
        if self._min_interval > 0:
            async with self._lock:
                now = time.monotonic()
                wait = max(0.0, self._next_slot - now)
                self._next_slot = max(now, self._next_slot) + self._min_interval
            if wait > 0:
                await asyncio.sleep(wait)

    def release(self) -> None:
        self._sem.release()
