"""分析层：聚合 / 对比 / case 挖掘 / 优化建议。"""
from .advisor import advise, weaknesses
from .aggregate import model_overview, overview_by_slice, per_model
from .cases import disagreement_cases, focal_vs_competitors
from .compare import dim_gap, pairwise_by_category, pairwise_winrate

__all__ = [
    "model_overview",
    "overview_by_slice",
    "per_model",
    "pairwise_winrate",
    "pairwise_by_category",
    "dim_gap",
    "focal_vs_competitors",
    "disagreement_cases",
    "weaknesses",
    "advise",
]
