"""Skill 注册中心 - 管理所有 Skill 的注册和发现。

此模块提供 SkillRegistry 类，用于统一管理所有 Skill。
"""

import logging
from typing import Dict, List, Optional
from .base import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Skill 注册中心。
    
    负责注册、发现和管理所有可用的 Skill，并为主图路由器
    动态生成技能描述 prompt。
    """
    
    def __init__(self):
        """初始化注册中心。"""
        self._skills: Dict[str, BaseSkill] = {}
        logger.info("Skill registry initialized")
    
    def register(self, skill: BaseSkill) -> None:
        """注册一个 Skill。"""
        if skill.name in self._skills:
            logger.warning(f"Skill '{skill.name}' already registered, overwriting")
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")
    
    def unregister(self, skill_name: str) -> None:
        """注销一个 Skill。"""
        if skill_name in self._skills:
            del self._skills[skill_name]
            logger.info(f"Unregistered skill: {skill_name}")
        else:
            logger.warning(f"Skill '{skill_name}' not found in registry")
    
    def get(self, skill_name: str) -> Optional[BaseSkill]:
        """获取一个 Skill。"""
        return self._skills.get(skill_name)
    
    def list_skills(self) -> List[str]:
        """列出所有已注册的 Skill 名称。"""
        return list(self._skills.keys())
    
    def get_all(self) -> Dict[str, BaseSkill]:
        """获取所有已注册的 Skill（name -> skill）。"""
        return self._skills.copy()
    
    def get_metadata(self) -> List[Dict[str, str]]:
        """获取所有 Skill 的元数据列表。"""
        return [skill.get_metadata() for skill in self._skills.values()]

    def build_router_prompt(self) -> str:
        """将所有已注册 Skill 的描述拼装为路由器 system prompt 片段。

        从每个 Skill 的 description（来源于 SKILL.md 的 ## 目的 + ## 适用场景）
        生成结构化描述，供主图 query_router_node 的 LLM 做分类决策。

        返回格式示例：
            【simple_query】
            **目的**
            处理单表或简单多表 SQL 查询...

            **适用场景**
            - 单表查询
            ...

            【complex_query】
            ...
        """
        blocks = []
        for name, skill in self._skills.items():
            desc = skill.description.strip() if skill.description else f"Skill: {name}"
            blocks.append(f"【{name}】\n{desc}")
        return "\n\n".join(blocks)

