"""OpenAI 兼容 runner —— 豆包(火山 Ark)及其他兼容厂商走这个。

火山 Ark：base_url=https://ark.cn-beijing.volces.com/api/v3，api_key=$ARK_API_KEY，model=endpoint_id。
"""
from __future__ import annotations

from .base import BaseRunner


class OpenAICompatRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.base_url:
            raise ValueError(f"openai_compat runner[{cfg.name}] 缺少 base_url")
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key() or "EMPTY")
        self.model = cfg.model or cfg.name

    async def _call(self, prompt: str, **kw) -> dict:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens or 4096,
        )
        choice = resp.choices[0]
        answer = (getattr(choice.message, "content", "") or "").strip()
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) if usage else 0
        tout = getattr(usage, "completion_tokens", 0) if usage else 0
        return {
            "answer": answer,
            "tokens_in": tin,
            "tokens_out": tout,
            "raw": {"model": self.model, "finish_reason": getattr(choice, "finish_reason", None)},
        }
