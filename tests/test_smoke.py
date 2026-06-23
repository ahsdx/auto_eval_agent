"""端到端 smoke：FuncRunner(假 agent) + FakeJudge(假裁判) 跑通全链路。"""
from pathlib import Path

import pytest

from auto_eval.batch import Orchestrator
from auto_eval.config import (
    AppConfig,
    EnsembleConfig,
    EvalOptions,
    JudgeConfig,
    ModelConfig,
    RubricDim,
)
from auto_eval.dataset import load_dataset
from auto_eval.engine import EvalEngine
from auto_eval.report import build_reports

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl"


def _cfg() -> AppConfig:
    return AppConfig(
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


@pytest.mark.asyncio
async def test_end_to_end(tmp_path, fake_agent_module, patch_judge):
    cfg = _cfg()
    items = load_dataset(str(DATA))
    run_dir = tmp_path / "run"

    # ① 批跑（FuncRunner 调假 agent）
    orch = Orchestrator(run_dir)
    answers = await orch.run(items, cfg.models)
    assert set(answers.keys()) == {"my_agent", "doubao"}
    assert all(len(v) == len(items) for v in answers.values())

    # ② 盲评 + 元评测（FakeJudge）
    engine = EvalEngine(cfg, run_dir, global_concurrency=8)
    results = await engine.evaluate(items, answers, focal_model="my_agent")
    assert len(results.verdicts) == len(items) * 2  # 每题每模型一个
    assert len(results.metas) == len(items) * 2
    # 无参考答案题应有成对比较
    noref = [it for it in items if not it.has_ref]
    if noref:
        assert len(results.pairs) == len(noref)  # focal vs 每个竞品

    # ③ 报告
    ans_index = {m: {o.item_id: o for o in outs} for m, outs in answers.items()}
    reps = build_reports(results, items, ans_index, cfg, "test-run")
    assert "A_md" in reps and "B_md" in reps
    assert "待评测" in reps["A_md"] or "对比报告" in reps["A_md"]
    assert "可靠性" in reps["B_md"]


@pytest.mark.asyncio
async def test_checkpoint_resume(tmp_path, fake_agent_module, patch_judge):
    """断点续跑：第二次运行应跳过已完成的题。"""
    cfg = _cfg()
    items = load_dataset(str(DATA))[:3]
    run_dir = tmp_path / "run"
    orch = Orchestrator(run_dir)
    await orch.run(items, cfg.models)
    first = (run_dir / "answers" / "my_agent.jsonl").read_text(encoding="utf-8").count("\n")
    # 再跑一次同样数据，不应新增行
    await orch.run(items, cfg.models)
    second = (run_dir / "answers" / "my_agent.jsonl").read_text(encoding="utf-8").count("\n")
    assert first == second == 3
