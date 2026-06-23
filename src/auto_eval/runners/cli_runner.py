"""命令行进程 runner —— subprocess 调起自研 agent，stdin 传 prompt、stdout 取回答。"""
from __future__ import annotations

import asyncio

from .base import BaseRunner


class CliRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.command:
            raise ValueError(f"cli runner[{cfg.name}] 缺少 command")

    async def _call(self, prompt: str, **kw) -> dict:
        proc = await asyncio.create_subprocess_exec(
            *self.cfg.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(prompt.encode("utf-8"))
        text = out.decode("utf-8", errors="ignore").strip()
        raw = {
            "exit": proc.returncode,
            "stderr": err.decode("utf-8", errors="ignore")[:500],
        }
        if proc.returncode != 0:
            raise RuntimeError(f"cli 退出码 {proc.returncode}: {raw['stderr']}")
        return {"answer": text, "raw": raw}
