"""HTTP 服务 runner —— 适配任意 HTTP 形态的自研 agent。"""
from __future__ import annotations

import httpx

from .base import BaseRunner


def _extract(data, path: str):
    """简化 jsonpath：支持 $.a.b 或 a.b 或 $.a[0].b。"""
    if not path:
        return data
    p = path.lstrip("$").lstrip(".")
    cur = data
    for seg in p.split("."):
        if seg == "":
            continue
        if "[" in seg:  # 形如 data[0]
            key, _, idx = seg.partition("[")
            if key and isinstance(cur, dict):
                cur = cur.get(key)
            if cur is None:
                return None
            idx = idx.rstrip("]")
            cur = cur[int(idx)] if idx.isdigit() else None
        elif isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            cur = None
        if cur is None:
            return None
    return cur


class HttpRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.url:
            raise ValueError(f"http runner[{cfg.name}] 缺少 url")
        self.client = httpx.AsyncClient(timeout=120.0)
        self.answer_path = cfg.answer_jsonpath

    async def _call(self, prompt: str, **kw) -> dict:
        body = {self.cfg.prompt_field: prompt, **self.cfg.extra}
        r = await self.client.request(
            self.cfg.method, self.cfg.url, json=body, headers=self.cfg.headers
        )
        r.raise_for_status()
        data = r.json()
        answer = _extract(data, self.answer_path)
        if not isinstance(answer, str):
            answer = "" if answer is None else str(answer)
        return {"answer": answer.strip(), "raw": data}

    async def aclose(self):
        await self.client.aclose()
