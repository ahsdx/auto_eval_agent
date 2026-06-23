"""探测脚本：验证加固后的盲评 agent loop —— 多工具、强制事实核查、查证轨迹。

评三道"答案含不同类型错误"的题，分别触发 web_search / calculate / python_run：
  1. 事实题（答案写错人）→ web_search
  2. 计算题（答案算错）  → calculate
  3. 编程题（代码缺终止条件）→ python_run

运行：python probe_agent_loop.py
"""
import asyncio
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from auto_eval.config import JudgeConfig, RubricDim  # noqa: E402
from auto_eval.judges import RubricJudge  # noqa: E402
from auto_eval.judges.base import JudgeClient  # noqa: E402
from auto_eval.schema import EvalItem  # noqa: E402


async def main():
    if not os.environ.get("KIMI_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        print("✗ 需要 .env 里的 KIMI_API_KEY 和 TAVILY_API_KEY")
        return

    cfg = JudgeConfig(
        name="kimi_judge",
        runner="openai_compat",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        model="moonshot-v1-8k",
        persona="strict_expert",
        enable_web_search=True,
        enable_fetch=True,
        enable_calculate=True,
        enable_python=True,
        temperature=0,
    )
    client = JudgeClient(cfg, search_provider="tavily", search_topk=3, max_rounds=8)
    judge = RubricJudge(
        client,
        [RubricDim(name=n, description="d", scale=5) for n in ["准确性", "完整性", "相关性", "有用性", "安全性"]],
    )

    cases = [
        (
            EvalItem(id="probe_fact", question="2024 年的诺贝尔文学奖获得者是谁？请简要说明。",
                     has_ref=False, category="事实", difficulty="medium"),
            "2024 年的诺贝尔文学奖得主是日本作家村上春树，表彰他对当代文学的贡献。",
        ),
        (
            EvalItem(id="probe_calc", question="计算 17 × 24 等于多少？",
                     has_ref=False, category="计算", difficulty="easy"),
            "17 × 24 = 410。",
        ),
        (
            EvalItem(id="probe_code", question="下面这个计算阶乘的 Python 函数是否正确？\ndef fact(n): return n * fact(n-1)",
                     has_ref=False, category="编程", difficulty="medium"),
            "这个函数是正确的，可以正确计算任意正整数的阶乘。",
        ),
    ]

    for i, (item, answer) in enumerate(cases, 1):
        print(f"\n{'='*70}\n【题 {i}】{item.question}\n待评答案：{answer}\n▶ 裁判 agent loop …")
        score = await judge.score(item, "under_test", answer)
        print("-" * 70)
        print(f"查证轨迹（{len(score.tool_trace)} 次工具调用）：")
        for t in score.tool_trace:
            print(f"  - {t}")
        print(f"判定：{score.correctness}   归因：{score.error_type}   总分：{score.total}")
        print(f"理由：{score.rationale}")


if __name__ == "__main__":
    asyncio.run(main())
