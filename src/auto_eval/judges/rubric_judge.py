"""Rubric 盲打分裁判：意图理解 + 理想锚定 + 多角度分析的深度盲评（可联网）。"""
from __future__ import annotations

import time
from datetime import datetime

from ..config import RubricDim
from ..schema import EvalItem, SingleScore
from .base import JudgeClient
from .prompts import (
    RUBRIC_PROCESS_SYSTEM,
    RUBRIC_PROCESS_USER,
    RUBRIC_SYSTEM,
    RUBRIC_USER,
    parse_analysis,
    parse_json_loose,
)

_VALID = {"right", "wrong", "partial", "unclear"}
_CLASSIFY_LABELS = "创作 / 事实 / 计算 / 翻译 / 建议"
_CLASSIFY_SYSTEM = f"你是查询分类器。只输出一个分类标签（{_CLASSIFY_LABELS}），不要任何其他文字。"


def _flatten_rubric(raw, dim_names=None):
    """将嵌套（一级→二级）rubric 展平为一级分 dict。兼容旧平坦格式。
    若 dim_names 提供且裁判输出的 key 集与期望维度名不匹配，则按顺序重映射
    （裁判通常按 prompt 列出的顺序输出，只是 key 名可能自创）。"""
    out = {}
    for k, v in (raw or {}).items():
        if isinstance(v, dict):
            out[k] = round(sum(v.values()) / len(v)) if v else 0
        elif isinstance(v, (int, float)):
            out[k] = int(v)
    if dim_names and out and set(out.keys()) != set(dim_names):
        vals = list(out.values())
        if len(vals) == len(dim_names):
            out = {dim_names[i]: vals[i] for i in range(len(dim_names))}
    return out


class RubricJudge:
    def __init__(self, client: JudgeClient, dims: list[RubricDim], skill_router=None):
        self.client = client
        self.dims = dims
        self.scale = dims[0].scale if dims else 5
        self.skill_router = skill_router

    async def score(self, item: EvalItem, model_name: str, answer: str, run_idx: int = 0,
                    eval_mode: str = "result", process_dims=None) -> SingleScore:
        today = datetime.now().strftime("%Y年%m月%d日")
        # 自动垂域分类：未标记 category 时让裁判分析 query 类型
        if not item.category or item.category == "default":
            label = await self._classify(item)
            if label:
                item.category = label
        skill_dims, skill_rules, _ = (self.skill_router.match(item) if self.skill_router else (None, "", []))
        if eval_mode == "process" and process_dims and item.trace:
            dims = process_dims  # 过程盲评维度不变（不受垂域 skill 影响）
            system = RUBRIC_PROCESS_SYSTEM.render(
                persona=self.client.persona, scale=dims[0].scale if dims else 5, dims=dims, skill_rules=skill_rules
            )
            user = RUBRIC_PROCESS_USER.render(
                question=item.question, context=item.context, answer=answer, trace=item.trace, current_date=today
            )
        else:
            dims = skill_dims or self.dims  # 垂域 skill 维度优先
            system = RUBRIC_SYSTEM.render(
                persona=self.client.persona, scale=self.scale, dims=dims, skill_rules=skill_rules
            )
            user = RUBRIC_USER.render(
                question=item.question, context=item.context, model_name=model_name, answer=answer, current_date=today
            )
        t0 = time.perf_counter()
        reply = await self.client.complete(system, user)
        latency = int((time.perf_counter() - t0) * 1000)

        analysis = parse_analysis(reply.content)
        data = parse_json_loose(reply.content) or {}
        rubric_raw = data.get("rubric") or {}
        rubric = _flatten_rubric(rubric_raw, dim_names=[d.name for d in dims])
        if data.get("total") is not None:
            total = float(data["total"])
        else:
            total = sum(rubric.values()) / len(rubric) if rubric else 0.0
        correctness = data.get("correctness", "unclear")
        if correctness not in _VALID:
            correctness = "unclear"
        error_type = data.get("error_type")
        rationale = data.get("rationale", "")

        return SingleScore(
            item_id=item.id,
            model=model_name,
            judge=self.client.cfg.name,
            persona=self.client.cfg.persona,
            run_idx=run_idx,
            rubric=rubric,
            total=total,
            correctness=correctness,
            error_type=error_type,
            rationale=rationale,
            analysis=analysis,
            used_search=reply.used_search,
            tool_trace=reply.tool_trace,
            search_queries=reply.search_queries,
            truncated=reply.truncated,
            latency_ms=latency,
        )

    async def _classify(self, item: EvalItem) -> str | None:
        """轻量垂域分类：未标记 category 时，让裁判分析 query 类型。只用一次 LLM 调用，不调工具。
        max_tokens=50 防止推理模型被 reasoning 吃光 token。"""
        try:
            resp = await self.client.client.chat.completions.create(
                model=self.client.model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user", "content": f"查询：{item.question}\n\n标签："},
                ],
                temperature=0,
                max_tokens=50,
            )
            label = (resp.choices[0].message.content or "").strip()
            # 容错：取第一个中文词，非标准标签返回 None（走 default）
            valid = {"创作", "事实", "计算", "翻译", "建议"}
            for v in valid:
                if v in label:
                    return v
            return label if label else None
        except Exception:
            return None
