"""客观真值：用参考答案算 EM / F1 / 语义相似度 → 客观对错。

仅在元评测阶段使用（校验评测 agent），不进入作答与盲评。
语义相似度这里用字符级 F1 作为无外部依赖的代理；可替换为 embedding。
"""
from __future__ import annotations

import re
from collections import Counter

_PUNCT = r"[，。！？；：、“”‘’（）【】《》,.!?;:\"'()\[\]{}<>·…—\-+/\\|*`~@#$%^&=]"


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(_PUNCT, "", s)
    return s


def tokens(s: str) -> list[str]:
    # 字符级，适合中英混排
    return list(normalize(s))


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def similarity(pred: str, ref: str) -> float:
    """语义相似度代理（字符级 F1）。可替换为 embedding cosine。"""
    return f1(pred, ref)


def compute(pred: str, ref: str) -> dict:
    em = exact_match(pred, ref)
    score = f1(pred, ref)
    if em == 1.0 or score >= 0.85:
        oc = "right"
    elif score >= 0.4:
        oc = "partial"
    else:
        oc = "wrong"
    return {"em": em, "f1": score, "sim": score, "objective_correct": oc}
