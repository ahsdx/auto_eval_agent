"""评估执行：分发三种模式 + 并发 + 推 SSE 事件 + 元评测汇总。

复用 auto_eval 核心：RubricJudge / PairwiseJudge / aggregate_* / build_runner / ground_truth。
"""
from __future__ import annotations

import asyncio

from ..config import AppConfig
from ..dataset import to_prompt
from ..judges import Arbitrator, JudgeClient, PairwiseJudge, RubricJudge, SkillRouter
from ..judges.ensemble import aggregate_pairs, aggregate_scores
from ..meta import ground_truth
from ..runners import build_runner
from ..schema import EvalItem
from .tasks import Task


def _to_evalitem(item: dict, idx: int) -> EvalItem:
    return EvalItem(
        id=item.get("id", f"q{idx}"),
        question=item["query"],
        context=item.get("context"),
        has_ref=bool(item.get("reference")),
        reference=item.get("reference"),
        category=item.get("category", "default"),
        trace=item.get("trace"),
    )


async def run_eval(task: Task, cfg: AppConfig) -> None:
    await task.publish("start", {"total": len(task.items), "mode": task.mode})
    task.status = "running"
    try:
        await _run(task, cfg)
        task.summary = _summarize(task, cfg.rubrics[0].scale if cfg.rubrics else 5)
        task.status = "done"
        await task.publish("done", {"summary": task.summary, "total": len(task.items)})
    except Exception as e:
        task.status = "error"
        task.error = f"{type(e).__name__}: {e}"
        await task.publish("error", {"message": task.error})


async def _run(task: Task, cfg: AppConfig) -> None:
    selected = task.options.get("judges") or [cfg.judges[0].name]
    judges_cfg = [j for j in cfg.judges if j.name in selected] or cfg.judges[:1]
    clients = [
        JudgeClient(j, cfg.eval_options.search_provider, cfg.eval_options.search_topk)
        for j in judges_cfg
    ]
    skill_router = SkillRouter(cfg.domain_skills) if cfg.domain_skills else None
    rubrics = [RubricJudge(c, cfg.rubrics, skill_router) for c in clients]
    pair_judges = [PairwiseJudge(c) for c in clients]
    scale = cfg.rubrics[0].scale if cfg.rubrics else 5
    sem = asyncio.Semaphore(int(task.options.get("concurrency", 4)))

    online_runner = None
    if task.mode == "online":
        model_name = task.options.get("model") or cfg.models[0].name
        mc = next((m for m in cfg.models if m.name == model_name), cfg.models[0])
        online_runner = build_runner(mc)
    process_dims = cfg.process_rubrics
    arbitrator = Arbitrator(clients[0]) if len(judges_cfg) >= 2 else None

    async def one(idx: int, item_dict: dict):
        async with sem:
            try:
                res = await _eval_one(
                    task.mode, idx, item_dict, rubrics, pair_judges, cfg, scale, online_runner, process_dims, arbitrator
                )
            except Exception as e:
                res = {
                    "index": idx,
                    "query": item_dict.get("query", ""),
                    "error": f"{type(e).__name__}: {e}",
                }
        res["index"] = idx
        task.results.append(res)
        task.done_total += 1
        await task.publish(
            "result",
            {"progress": task.done_total, "total": len(task.items), "result": res},
        )

    await asyncio.gather(*[one(i, it) for i, it in enumerate(task.items)])


async def _eval_one(mode, idx, item_dict, rubrics, pair_judges, cfg, scale, online_runner, process_dims=None, arbitrator=None) -> dict:
    item = _to_evalitem(item_dict, idx)
    out: dict = {"query": item.question}
    if item_dict.get("category"):
        out["category"] = item_dict["category"]

    if mode in ("single", "process"):
        answer = item_dict["answer"]
        out["answer"] = answer
        if mode == "process":
            out["trace"] = (item.trace or "")[:200]
            eval_mode, dims = "process", (process_dims or cfg.rubrics)
        else:
            eval_mode, dims = "result", cfg.rubrics
        scores = await asyncio.gather(*[
            r.score(item, "answer", answer, eval_mode=eval_mode, process_dims=process_dims) for r in rubrics
        ])
        v = aggregate_scores(list(scores), dims, cfg.ensemble, cfg.ensemble.flag_low_agreement)
        # 多裁判分歧 → 主席仲裁（覆盖为主席最终结论）
        if v and v.low_agreement and len(scores) >= 2 and arbitrator:
            try:
                arb = await arbitrator.arbitrate(item, answer, list(scores))
                v.correctness, v.total, v.rubric = arb["correctness"], arb["total"], arb["rubric"]
                v.arbitrated = True
                v.arbitrator_confidence = arb["confidence"]
                v.arbitrator_rationale = arb["rationale"]
                v.rationale = f"[主席仲裁·置信度{arb['confidence']}] {arb['rationale']}"
            except Exception:
                pass
        _fill_verdict(out, v)
        _maybe_meta(out, item, answer, v)

    elif mode == "compare":
        aa, ab = item_dict["answer_a"], item_dict["answer_b"]
        out["answer_a"], out["answer_b"] = aa, ab
        pairs = []
        for pj in pair_judges:
            pairs.append(await pj.compare_once(item, "A", aa, "B", ab, order="ab"))
            if cfg.eval_options.pairwise_bidirectional:
                pairs.append(await pj.compare_once(item, "A", aa, "B", ab, order="ba"))
        pr = aggregate_pairs(pairs, cfg.ensemble, cfg.ensemble.flag_low_agreement)
        if pr is None:
            out["error"] = "裁判无成对输出"
        else:
            out.update(
                winner=pr.winner, a_wins=pr.a_wins, b_wins=pr.b_wins, ties=pr.ties,
                bidirectional_consistent=pr.bidirectional_consistent,
                rationale=pr.rationale, low_agreement=pr.low_agreement,
            )

    else:  # online
        mo = await online_runner.generate_strict(to_prompt(item), item_id=item.id)
        out["generated_answer"] = mo.answer
        out["answer"] = mo.answer
        if mo.error:
            out["gen_error"] = mo.error
        scores = await asyncio.gather(*[r.score(item, "answer", mo.answer) for r in rubrics])
        v = aggregate_scores(list(scores), cfg.rubrics, cfg.ensemble, cfg.ensemble.flag_low_agreement)
        _fill_verdict(out, v)
        _maybe_meta(out, item, mo.answer, v)

    return out


def _fill_verdict(out: dict, v) -> None:
    if v is None:
        out["error"] = out.get("error", "裁判无输出")
        return
    out["correctness"] = v.correctness
    out["total"] = round(v.total, 2)
    out["rubric"] = {k: round(val, 2) for k, val in v.rubric.items()}
    out["error_type"] = v.error_type
    out["rationale"] = v.rationale
    out["tool_trace"] = v.single_scores[0].tool_trace if v.single_scores else []
    out["used_search"] = any(s.used_search for s in v.single_scores)
    out["truncated"] = any(s.truncated for s in v.single_scores)
    out["low_agreement"] = v.low_agreement
    out["arbitrated"] = v.arbitrated
    out["arbitrator_confidence"] = v.arbitrator_confidence


def _maybe_meta(out: dict, item: EvalItem, answer: str, v) -> None:
    if item.reference and v is not None:
        obj = ground_truth.compute(answer, item.reference)
        out["objective"] = obj
        out["agree"] = (v.correctness == obj["objective_correct"]) if v.correctness != "unclear" else None


def _summarize(task: Task, scale: int) -> dict:
    res = task.results
    ok = [r for r in res if "error" not in r]
    summary: dict = {"total": len(res), "done": len(ok), "mode": task.mode}
    if task.mode in ("single", "online"):
        totals = [r.get("total") for r in ok if r.get("total") is not None]
        if totals:
            summary["mean_total"] = round(sum(totals) / len(totals), 2)
            summary["norm_mean"] = round(sum(totals) / len(totals) / scale, 3)
        has_meta = [r for r in ok if "agree" in r]
        if has_meta:
            agreed = sum(1 for r in has_meta if r.get("agree") is True)
            summary["meta_n"] = len(has_meta)
            summary["judge_accuracy"] = round(agreed / len(has_meta), 3)
    elif task.mode == "compare":
        a = sum(r.get("a_wins", 0) for r in ok)
        b = sum(r.get("b_wins", 0) for r in ok)
        t = sum(r.get("ties", 0) for r in ok)
        tot = a + b + t
        summary["a_winrate"] = round((a + 0.5 * t) / tot, 3) if tot else None
    return summary
