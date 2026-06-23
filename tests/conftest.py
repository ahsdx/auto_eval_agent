"""测试公共设施：fake agent 模块 + fake 裁判客户端（不依赖真实 LLM/网络）。"""
import json
import sys
import types

import pytest

DATA = (lambda: None)  # placeholder


@pytest.fixture
def fake_agent_module():
    """注册一个可被 FuncRunner import 的假 agent 模块。"""
    mod = types.ModuleType("_fake_agent")

    def echo(prompt: str) -> str:
        return f"模拟答案：{prompt[:30]}"

    def echo2(prompt: str) -> str:
        return f"另一种回答：{prompt[:20]}"

    mod.echo = echo
    mod.echo2 = echo2
    sys.modules["_fake_agent"] = mod
    yield mod
    sys.modules.pop("_fake_agent", None)


class FakeJudgeClient:
    """假裁判：rubric 固定判 right，pairwise 固定判 a 胜。"""

    def __init__(self, cfg, *a, **kw):
        self.cfg = cfg
        from auto_eval.judges.prompts import persona_text

        self.persona = persona_text(cfg.persona)
        self.model = cfg.model or cfg.name
        self.has_tools = False

    async def complete(self, system: str, user: str):
        from auto_eval.judges.base import JudgeReply

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


@pytest.fixture
def patch_judge(monkeypatch):
    import auto_eval.engine as eng

    monkeypatch.setattr(eng, "JudgeClient", FakeJudgeClient)


@pytest.fixture
def fake_judge_cls():
    return FakeJudgeClient
