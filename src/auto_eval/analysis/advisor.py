"""弱点聚类 + 优化建议。

弱点按 (error_type × category) 聚类（无需 sklearn）；建议默认走规则模板，
可选传入裁判客户端用 LLM 增强（第二阶段）。
"""
from __future__ import annotations

import collections

from ..schema import EvalItem, Verdict


def weaknesses(verdicts, items_map: dict[str, EvalItem], focal: str) -> list[dict]:
    """focal 被判 wrong/partial 的题，按 (error_type, category) 聚类。"""
    bad = [v for v in verdicts.values() if v.model == focal and v.correctness in ("wrong", "partial")]
    clusters: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for v in bad:
        item = items_map.get(v.item_id)
        cat = item.categories()[0] if item and item.categories() else "default"
        et = v.error_type or "未归类"
        clusters[(et, cat)].append(v.item_id)
    out = []
    for (et, cat), ids in sorted(clusters.items(), key=lambda x: -len(x[1])):
        out.append({"error_type": et, "category": cat, "count": len(ids), "item_ids": ids[:10]})
    return out


def advise(
    focal: str,
    focal_overview: dict,
    weakness_list: list[dict],
    winrate_vs: dict[str, dict],
    scale: int,
) -> str:
    """规则模板生成优化建议（Markdown）。"""
    lines: list[str] = [f"# {focal} 优化建议", ""]

    # 1) 与竞品对比
    lines.append("## 一、与竞品整体对比")
    if not winrate_vs:
        lines.append("- 未生成成对比较结果（可能所有题都有参考答案且未开启 pairwise_for_ref）。")
    for other, wr in winrate_vs.items():
        rate = wr.get("focal_winrate")
        if rate is None:
            continue
        if rate < 0.45:
            lines.append(f"- ⚠️ 整体落后于 **{other}**（胜率 {rate:.1%}），建议作为重点优化方向。")
        elif rate > 0.55:
            lines.append(f"- ✅ 整体优于 {other}（胜率 {rate:.1%}）。")
        else:
            lines.append(f"- ➖ 与 {other} 基本持平（胜率 {rate:.1%}）。")
        if wr.get("bidirectional_inconsistent"):
            lines.append(f"  - 注意：{wr['bidirectional_inconsistent']} 题双向比较不一致（存在位置偏差或难度临界）。")
    lines.append("")

    # 2) 薄弱点聚类
    lines.append("## 二、薄弱点（按错误类型聚类）")
    if not weakness_list:
        lines.append("- 未发现明显错题，盲评表现良好。")
    for w in weakness_list[:10]:
        lines.append(
            f"- 【{w['error_type']} / {w['category']}】共 {w['count']} 题 —— "
            f"建议：针对性调整 Prompt 约束、补充相关检索/知识、或在训练数据中增加该类样本。"
        )
    lines.append("")

    # 3) 维度短板
    lines.append("## 三、评分维度短板")
    dims = focal_overview.get("dim_means", {})
    if dims:
        threshold = 0.6 * scale
        weak_dims = sorted(dims.items(), key=lambda x: x[1])[:5]
        for d, val in weak_dims:
            flag = " ⚠️偏低" if val < threshold else ""
            lines.append(f"- {d}：{val:.2f} / {scale}{flag}")
    lines.append("")

    # 4) 评测可靠性提示
    low_rate = focal_overview.get("low_agreement_rate", 0)
    if low_rate > 0.2:
        lines.append("## 四、评测可靠性提示")
        lines.append(
            f"- {focal} 有 {low_rate:.1%} 的题多裁判一致率偏低，相关结论建议人工复核后再决策。"
        )

    return "\n".join(lines)
