"""生成示例报告的 demo：假 agent + 假裁判，无需任何 API key。

运行：python demo_run.py
产物：runs/demo/reports/A.md, A.json, B.md, B.json
"""
import asyncio
import json
import sys
import types
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows 控制台默认 GBK
except Exception:
    pass

# 注册假 agent 模块（FuncRunner 会 import 它）
_mod = types.ModuleType("_fake_agent")
_mod.echo = lambda p: f"模拟答案：{p[:25]}"
_mod.echo2 = lambda p: f"另一种回答：{p[:18]}"
sys.modules["_fake_agent"] = _mod

# 假裁判（覆盖 JudgeClient）
from auto_eval.engine import EvalEngine  # noqa: E402
import auto_eval.engine as _eng  # noqa: E402
from auto_eval.judges.base import JudgeReply  # noqa: E402


class FakeJudgeClient:
    def __init__(self, cfg, *a, **kw):
        self.cfg = cfg
        from auto_eval.judges.prompts import persona_text

        self.persona = persona_text(cfg.persona)
        self.model = cfg.model or cfg.name
        self.has_tools = False

    async def complete(self, system, user):
        if "答案 A" in user and "答案 B" in user:
            return JudgeReply(content=json.dumps({"winner": "a", "rationale": "fake: A 略好"}))
        return JudgeReply(
            content=json.dumps(
                {
                    "rubric": {"准确性": 4, "完整性": 4, "相关性": 5, "有用性": 4, "安全性": 5},
                    "total": 4.4,
                    "correctness": "right",
                    "error_type": None,
                    "rationale": "fake: 基本正确",
                }
            )
        )


_eng.JudgeClient = FakeJudgeClient

from auto_eval.batch import Orchestrator  # noqa: E402
from auto_eval.config import (  # noqa: E402
    AppConfig,
    EnsembleConfig,
    EvalOptions,
    JudgeConfig,
    ModelConfig,
    RubricDim,
)
from auto_eval.dataset import load_dataset  # noqa: E402
from auto_eval.report import build_reports  # noqa: E402

BASE = Path(__file__).resolve().parent
CFG = AppConfig(
    models=[
        ModelConfig(name="my_agent", runner="func", func_module="_fake_agent:echo", concurrency=2),
        ModelConfig(name="doubao", runner="func", func_module="_fake_agent:echo2", concurrency=2),
    ],
    judges=[
        JudgeConfig(name="j1", base_url="http://x", model="m1", persona="strict_expert"),
        JudgeConfig(name="j2", base_url="http://x", model="m2", persona="end_user"),
    ],
    rubrics=[RubricDim(name=n, description="d", scale=5) for n in ["准确性", "完整性", "相关性", "有用性", "安全性"]],
    eval_options=EvalOptions(repeat=1, pairwise_bidirectional=True),
    ensemble=EnsembleConfig(),
)


async def main():
    items = load_dataset(str(BASE / "data" / "dataset.jsonl"))
    run_dir = BASE / "runs" / "demo"
    print(f"▶ 批跑 {len(items)} 题 × 2 模型 …")
    answers = await Orchestrator(run_dir).run(items, CFG.models)
    print("▶ 盲评 + 元评测 …")
    engine = EvalEngine(CFG, run_dir, global_concurrency=8)
    results = await engine.evaluate(items, answers, focal_model="my_agent")
    print("▶ 生成报告 …")
    ans_index = {m: {o.item_id: o for o in outs} for m, outs in answers.items()}
    reps = build_reports(results, items, ans_index, CFG, "demo")
    rep = run_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "A.md").write_text(reps["A_md"], encoding="utf-8")
    (rep / "A.json").write_text(json.dumps(reps["A"], ensure_ascii=False, indent=2), encoding="utf-8")
    (rep / "B.md").write_text(reps["B_md"], encoding="utf-8")
    (rep / "B.json").write_text(json.dumps(reps["B"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 示例报告：\n  {rep/'A.md'}\n  {rep/'B.md'}")


if __name__ == "__main__":
    asyncio.run(main())
