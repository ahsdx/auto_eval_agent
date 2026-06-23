"""真实裁判(Kimi+Tavily) + 假被测答案，跑示例集 8 题，生成含查证轨迹的真实报告。

被测答案用假 agent（回显），重点验证：真实裁判 agent loop 在多题上的盲评 +
元评测校验 + 报告里的查证轨迹。消耗 Kimi/Tavily 额度。

运行：python demo_real_judge.py
"""
import asyncio
import json
import sys
import types
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# 假被测 agent（两个略有不同的回显，模拟两个模型）
_mod = types.ModuleType("_fake_agent")
_mod.echo = lambda p: f"（my_agent 回答）{p[:40]}"
_mod.echo2 = lambda p: f"（doubao 回答）关于「{p[:24]}」，这是一个模拟回答。"
sys.modules["_fake_agent"] = _mod

from auto_eval.config import (  # noqa: E402
    AppConfig,
    EnsembleConfig,
    EvalOptions,
    JudgeConfig,
    ModelConfig,
    RubricDim,
)
from auto_eval.dataset import load_dataset  # noqa: E402
from auto_eval.batch import Orchestrator  # noqa: E402
from auto_eval.engine import EvalEngine  # noqa: E402
from auto_eval.report import build_reports  # noqa: E402

BASE = Path(__file__).resolve().parent
CFG = AppConfig(
    models=[
        ModelConfig(name="my_agent", runner="func", func_module="_fake_agent:echo", concurrency=2),
        ModelConfig(name="doubao", runner="func", func_module="_fake_agent:echo2", concurrency=2),
    ],
    judges=[
        JudgeConfig(
            name="judge_kimi",
            runner="openai_compat",
            base_url="https://api.moonshot.cn/v1",
            api_key_env="KIMI_API_KEY",
            model="moonshot-v1-8k",
            persona="strict_expert",
            enable_web_search=True,
            enable_fetch=True,
            enable_calculate=True,
            enable_python=True,
            concurrency=2,
        ),
    ],
    rubrics=[RubricDim(name=n, description="d", scale=5) for n in ["准确性", "完整性", "相关性", "有用性", "安全性"]],
    eval_options=EvalOptions(
        repeat=1, pairwise_bidirectional=True, search_provider="tavily", search_topk=3
    ),
    ensemble=EnsembleConfig(),
)


async def main():
    items = load_dataset(str(BASE / "data" / "dataset.jsonl"))
    run_dir = BASE / "runs" / "demo_real"
    print(f"▶ 批跑 {len(items)} 题 × 2 模型（假答案）…")
    answers = await Orchestrator(run_dir).run(items, CFG.models)
    print("▶ 真实裁判(Kimi+Tavily) 盲评 + 元评测 …（含联网查证，请稍候）")
    engine = EvalEngine(CFG, run_dir, global_concurrency=4)
    results = await engine.evaluate(items, answers, focal_model="my_agent")
    print("▶ 生成报告 …")
    ans_index = {m: {o.item_id: o for o in outs} for m, outs in answers.items()}
    reps = build_reports(results, items, ans_index, CFG, "demo_real")
    rep = run_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "A.md").write_text(reps["A_md"], encoding="utf-8")
    (rep / "A.json").write_text(json.dumps(reps["A"], ensure_ascii=False, indent=2), encoding="utf-8")
    (rep / "B.md").write_text(reps["B_md"], encoding="utf-8")
    (rep / "B.json").write_text(json.dumps(reps["B"], ensure_ascii=False, indent=2), encoding="utf-8")

    ms = reps["B"]["meta_summary"]
    print(f"\n✅ 报告：{rep/'A.md'} / {rep/'B.md'}")
    print(f"   元评测：有ref题 {ms['n_has_ref']} 道，裁判判题准确率 "
          f"{ms['judge_accuracy']*100:.0f}%，偏差 {ms['bias_direction']}")


if __name__ == "__main__":
    asyncio.run(main())
