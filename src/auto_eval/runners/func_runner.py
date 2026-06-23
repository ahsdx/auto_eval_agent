"""本地 Python 函数 runner —— 直接 import 自研 agent 调用。

配置 func_module 形如 'mypkg.agent:chat' 或 'mypkg.agent:Client.chat'。
被调用对象可同步或异步，返回 str 或 {'answer': ...}。
"""
from __future__ import annotations

import inspect

from .base import BaseRunner


def _load_callable(spec: str):
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("func_module 需为 'module.path:callable' 形式")
    import importlib

    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class FuncRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.func_module:
            raise ValueError(f"func runner[{cfg.name}] 缺少 func_module")
        self.fn = _load_callable(cfg.func_module)

    async def _call(self, prompt: str, **kw) -> dict:
        res = self.fn(prompt)
        if inspect.isawaitable(res):
            res = await res
        if isinstance(res, str):
            answer = res
        elif isinstance(res, dict):
            answer = res.get("answer", "") or res.get("text", "") or res.get("output", "")
        else:
            answer = str(res)
        return {"answer": (answer or "").strip(), "raw": {"func": self.cfg.func_module}}
