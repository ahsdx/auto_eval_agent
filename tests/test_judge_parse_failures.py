import pytest

from auto_eval.config import JudgeConfig, RubricDim
from auto_eval.judges.base import JudgeReply
from auto_eval.judges.pairwise_judge import PairwiseJudge
from auto_eval.judges.rubric_judge import RubricJudge
from auto_eval.schema import EvalItem


class InvalidJsonClient:
    def __init__(self):
        self.cfg = JudgeConfig(name="invalid", runner="openai_compat")
        self.persona = "test"

    async def complete(self, system: str, user: str) -> JudgeReply:
        return JudgeReply(content="这不是合法 JSON")


@pytest.mark.asyncio
async def test_rubric_judge_does_not_silently_fallback_to_unclear():
    judge = RubricJudge(
        InvalidJsonClient(),
        [RubricDim(name="准确性", description="是否准确", weight=1, scale=5)],
    )

    with pytest.raises(ValueError, match="无法解析"):
        await judge.score(EvalItem(id="q1", question="1+1=?"), "answer", "2")


@pytest.mark.asyncio
async def test_pairwise_judge_does_not_silently_fallback_to_tie():
    judge = PairwiseJudge(InvalidJsonClient())

    with pytest.raises(ValueError, match="无法解析"):
        await judge.compare_once(
            EvalItem(id="q1", question="哪个更好？"),
            "A",
            "回答A",
            "B",
            "回答B",
        )