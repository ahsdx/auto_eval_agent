"""可重试异常与重试调用。"""
from __future__ import annotations

import asyncio

try:
    import httpx

    _HTTP = (httpx.HTTPError, httpx.TimeoutException)
except Exception:  # pragma: no cover - httpx 一定在依赖里
    _HTTP = ()

# 可重试：网络/超时/连接类。4xx（除 429）通常不可重试，会抛 HTTPStatusError——视情况。
RETRIABLE = _HTTP + (TimeoutError, ConnectionError, OSError)


async def retry_call(coro_factory, *, max_attempts: int = 4, base_wait: float = 1.0, max_wait: float = 30.0):
    """对 `coro_factory()`（返回协程的零参函数）做指数退避重试，仅重试 RETRIABLE 异常。"""
    last_exc = None
    for i in range(max_attempts):
        try:
            return await coro_factory()
        except RETRIABLE as e:
            last_exc = e
            if i == max_attempts - 1:
                break
            await asyncio.sleep(min(max_wait, base_wait * (2**i)))
    raise last_exc  # type: ignore[misc]
