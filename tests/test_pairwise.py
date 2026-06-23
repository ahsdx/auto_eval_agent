import pytest

from auto_eval.config import JudgeConfig
from auto_eval.judges import PairwiseJudge
from auto_eval.schema import EvalItem


@pytest.mark.asyncio
async def test_pairwise_order_normalize(fake_judge_cls):
    """ba 方向时，裁判说 a(呈现A好) 应归一化为固定 model_b 胜。"""
    item = EvalItem(id="x", question="q", has_ref=False, category="c")
    client = fake_judge_cls(JudgeConfig(name="j", base_url="http://x", model="m"))
    pj = PairwiseJudge(client)

    r_ab = await pj.compare_once(item, "A", "ansA", "B", "ansB", order="ab")
    r_ba = await pj.compare_once(item, "A", "ansA", "B", "ansB", order="ba")
    # fake 裁判恒返回 winner=a（呈现A更好）
    assert r_ab.winner == "a"  # 呈现A=model_a -> a
    assert r_ba.winner == "b"  # 呈现A=model_b -> 归一化后 model_b 胜 = b
