"""不调 LLM 的重报脚本：从已有 run 的 verdicts/pairs/answers 重建并重生成报告。

用于在改了报告渲染后，快速重出报告而不重跑裁判。
运行：python rereport.py [run_dir]
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from auto_eval.config import load_config  # noqa: E402
from auto_eval.dataset import load_dataset  # noqa: E402
from auto_eval.engine import EvalResults  # noqa: E402
from auto_eval.meta import per_item  # noqa: E402
from auto_eval.report import build_reports  # noqa: E402
from auto_eval.schema import ModelOutput, PairResult, Verdict  # noqa: E402

BASE = Path(__file__).resolve().parent


def _read_jsonl(path: Path, cls):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(cls(**json.loads(line)))
        except Exception:
            continue
    return out


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "runs" / "demo_real"
    cfg = load_config(BASE / "config")
    items = load_dataset(str(BASE / "data" / "dataset.jsonl"))
    items_map = {it.id: it for it in items}
    scale = cfg.rubrics[0].scale if cfg.rubrics else 5

    answers = {}
    for f in sorted((run_dir / "answers").glob("*.jsonl")):
        answers[f.stem] = _read_jsonl(f, ModelOutput)
    ans_index = {m: {o.item_id: o for o in outs} for m, outs in answers.items()}

    verdicts = {}
    for v in _read_jsonl(run_dir / "verdicts.jsonl", Verdict):
        verdicts[(v.item_id, v.model)] = v
    pairs = {}
    for p in _read_jsonl(run_dir / "pairs.jsonl", PairResult):
        pairs[(p.item_id, p.model_a, p.model_b)] = p

    metas = []
    for (iid, model), v in verdicts.items():
        item = items_map.get(iid)
        if not item:
            continue
        out = ans_index.get(model, {}).get(iid)
        metas.append(per_item(v, item, out.answer if out else "", scale))

    results = EvalResults(verdicts=verdicts, pairs=pairs, metas=metas, focal_model=cfg.model_names()[0])
    reps = build_reports(results, items, ans_index, cfg, run_dir.name)
    rep = run_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "A.md").write_text(reps["A_md"], encoding="utf-8")
    (rep / "A.json").write_text(json.dumps(reps["A"], ensure_ascii=False, indent=2), encoding="utf-8")
    (rep / "B.md").write_text(reps["B_md"], encoding="utf-8")
    (rep / "B.json").write_text(json.dumps(reps["B"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 重报完成：{rep/'A.md'} / {rep/'B.md'}")
    n_trace = len(reps["A"].get("judge_trace_examples", []))
    print(f"   含查证轨迹的裁判示例：{n_trace} 条")


if __name__ == "__main__":
    main()
