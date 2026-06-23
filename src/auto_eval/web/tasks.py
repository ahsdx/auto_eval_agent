"""任务管理：内存存储 + asyncio.Queue 作 SSE 事件总线。"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    id: str
    mode: str
    items: list[dict]
    options: dict
    status: str = "pending"  # pending | running | done | error
    results: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)
    done_total: int = 0
    error: str | None = None

    async def publish(self, event: str, data: dict) -> None:
        await self.queue.put({"event": event, "data": data})


TASKS: dict[str, Task] = {}


def new_task(mode: str, items: list[dict], options: dict) -> Task:
    task_id = uuid.uuid4().hex[:12]
    t = Task(id=task_id, mode=mode, items=items, options=options)
    TASKS[task_id] = t
    return t


def get_task(task_id: str) -> Task | None:
    return TASKS.get(task_id)
