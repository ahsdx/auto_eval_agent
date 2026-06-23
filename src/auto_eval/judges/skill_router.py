"""Skill 路由：按 EvalItem.category 匹配垂域评测 Skill。

每个 Skill 自带 rubrics（一级+二级维度），匹配后直接返回——不依赖全局维度库。
"""
from __future__ import annotations

from ..config import DomainSkill, RubricDim
from ..schema import EvalItem


class SkillRouter:
    def __init__(self, domain_skills: dict[str, DomainSkill]):
        self.domain = domain_skills  # {skill_name: DomainSkill}

    def match(self, item: EvalItem) -> tuple[list[RubricDim], str, list[str]]:
        """返回 (维度列表, 规则文字, 示例列表)。无匹配回落 default。"""
        cats = item.categories()
        for name, skill in self.domain.items():
            if name == "default":
                continue
            if any(c in skill.matching_categories for c in cats):
                return skill.rubrics, skill.rules, list(skill.examples or [])
        default = self.domain.get("default")
        if default:
            return default.rubrics, default.rules or "", list(default.examples or [])
        return [], "", []
