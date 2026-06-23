"""裁判校准：盲评结论 vs 参考答案客观真值 → 评测 agent 可靠性。"""
from __future__ import annotations

import collections
from typing import Any

import numpy as np

from ..schema import EvalItem, MetaResult, Verdict
from . import ground_truth


def per_item(verdict: Verdict, item: EvalItem, pred: str, scale: int) -> MetaResult:
    cat = item.categories()[0] if item.categories() else "default"
    if item.has_ref and item.reference:
        obj = ground_truth.compute(pred, item.reference)
        oc = obj["objective_correct"]
        jc = verdict.correctness
        agree = (jc == oc) if jc in ("right", "wrong", "partial") else None
        judge_norm = (verdict.total / scale) if scale else None
        delta = (judge_norm - obj["sim"]) if judge_norm is not None else None
        objective = {k: v for k, v in obj.items() if k != "objective_correct"}
        return MetaResult(
            item_id=item.id,
            model=verdict.model,
            has_ref=True,
            category=cat,
            difficulty=item.difficulty,
            objective=objective,
            objective_correct=oc,
            judge_correctness=jc,
            judge_total=verdict.total,
            agree=agree,
            delta_total_vs_sim=delta,
        )
    return MetaResult(
        item_id=item.id,
        model=verdict.model,
        has_ref=False,
        category=cat,
        difficulty=item.difficulty,
        agree=None,
    )


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    a, b = np.array(x, float), np.array(y, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(results: list[MetaResult]) -> dict[str, Any]:
    """汇总评测 agent 在有参考答案题上的可靠性。"""
    has_ref = [m for m in results if m.has_ref and m.agree is not None]
    n = len(has_ref)
    accuracy = (sum(1 for m in has_ref if m.agree) / n) if n else None

    deltas = [m.delta_total_vs_sim for m in has_ref if m.delta_total_vs_sim is not None]
    bias = float(np.mean(deltas)) if deltas else None

    jt = [m.judge_total for m in has_ref if m.judge_total is not None]
    sm = [m.objective.get("sim", 0.0) for m in has_ref if m.judge_total is not None]
    corr = _pearson(jt, sm)

    disagreements = [
        {
            "item_id": m.item_id,
            "model": m.model,
            "category": m.category,
            "difficulty": m.difficulty,
            "objective_correct": m.objective_correct,
            "judge_correctness": m.judge_correctness,
            "agree": m.agree,
        }
        for m in has_ref if m.agree is False
    ]

    by_cat: dict[str, list[bool]] = collections.defaultdict(list)
    for m in has_ref:
        by_cat[m.category].append(bool(m.agree))
    cat_acc = {c: float(np.mean(v)) for c, v in by_cat.items()}

    by_diff: dict[str, list[bool]] = collections.defaultdict(list)
    for m in has_ref:
        by_diff[m.difficulty].append(bool(m.agree))
    diff_acc = {d: float(np.mean(v)) for d, v in by_diff.items()}

    # 偏差方向定性
    if bias is None:
        bias_dir = "未知"
    elif bias > 0.1:
        bias_dir = "偏松（盲评分普遍高于客观相似度）"
    elif bias < -0.1:
        bias_dir = "偏严（盲评分普遍低于客观相似度）"
    else:
        bias_dir = "基本中性"

    return {
        "n_has_ref": n,
        "judge_accuracy": accuracy,
        "bias_judge_vs_sim": bias,
        "bias_direction": bias_dir,
        "score_corr_judge_vs_sim": corr,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:50],
        "category_accuracy": cat_acc,
        "difficulty_accuracy": diff_acc,
    }


def summarize_noref(verdicts: dict, items_map: dict) -> dict[str, Any]:
    """无参考答案题的盲评可信度：基于多裁判一致率/稳定性与 low_agreement 占比。"""
    no_ref_items = {iid for iid, it in items_map.items() if not it.has_ref}
    rel = [v for (iid, _m), v in verdicts.items() if iid in no_ref_items]
    n = len(rel)
    if not n:
        return {"n_no_ref": 0}
    low = sum(1 for v in rel if v.low_agreement)
    agreements = [v.judges_agreement for v in rel if v.judges_agreement is not None]
    mean_agree = float(np.mean(agreements)) if agreements else None
    return {
        "n_no_ref": n,
        "mean_judges_agreement": mean_agree,
        "low_agreement_rate": low / n,
        "flagged_for_review": low,
    }
