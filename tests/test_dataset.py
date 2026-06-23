from pathlib import Path

from auto_eval.dataset import assert_no_reference_leak, load_dataset, split_by_ref, to_prompt

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl"


def test_load_and_split():
    items = load_dataset(str(DATA))
    assert len(items) >= 5
    with_ref, without_ref = split_by_ref(items)
    assert len(with_ref) >= 1, "应含无参考答案题"
    assert len(without_ref) >= 1, "应含有参考答案题"


def test_reference_isolation():
    """reference 绝不能出现在喂给模型/裁判的 prompt 里。"""
    items = load_dataset(str(DATA))
    for it in items:
        prompt = to_prompt(it)
        assert_no_reference_leak(prompt, it)
        if it.has_ref and it.reference:
            assert it.reference not in prompt
