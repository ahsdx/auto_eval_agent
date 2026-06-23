"""断点续跑：每模型一个 jsonl，增量 append + 启动读取已完成 item_id。"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import ModelOutput


class AnswerStore:
    def __init__(self, run_dir: Path, model: str):
        self.path = Path(run_dir) / "answers" / f"{model}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def done_ids(self) -> set[str]:
        """已成功完成（无 error 且有 answer）的 item_id 集合，启动时跳过。"""
        ids: set[str] = set()
        if not self.path.exists():
            return ids
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("error") is None and obj.get("answer"):
                    ids.add(obj["item_id"])
        return ids

    def append(self, out: ModelOutput) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(out.model_dump_json() + "\n")

    def load_all(self) -> list[ModelOutput]:
        res: list[ModelOutput] = []
        if not self.path.exists():
            return res
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    res.append(ModelOutput(**json.loads(line)))
                except Exception:
                    continue
        return res
