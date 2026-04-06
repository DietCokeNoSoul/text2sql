"""Skill 注册中心 - 管理所有 Skill 的注册和发现。

此模块提供 SkillRegistry 类，用于统一管理所有 Skill。
"""

import logging
from typing import Dict, List, Optional
from .base import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Skill 注册中心。
    
    负责注册、发现和管理所有可用的 Skill。
    """
    
    def __init__(self):
        """初始化注册中心。"""
        self._skills: Dict[str, BaseSkill] = {}
        logger.info("Skill registry initialized")
    
    def register(self, skill: BaseSkill) -> None:
        """注册一个 Skill。
        
        参数:
            skill: 要注册的 Skill 实例
        """
        if skill.name in self._skills:
            logger.warning(f"Skill '{skill.name}' already registered, overwriting")
        
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")
    
    def unregister(self, skill_name: str) -> None:
        """注销一个 Skill。
        
        参数:
            skill_name: 要注销的 Skill 名称
        """
        if skill_name in self._skills:
            del self._skills[skill_name]
            logger.info(f"Unregistered skill: {skill_name}")
        else:
            logger.warning(f"Skill '{skill_name}' not found in registry")
    
    def get(self, skill_name: str) -> Optional[BaseSkill]:
        """获取一个 Skill。
        
        参数:
            skill_name: Skill 名称
            
        返回:
            Skill 实例，如果不存在则返回 None
        """
        return self._skills.get(skill_name)
    
    def list_skills(self) -> List[str]:
        """列出所有已注册的 Skill 名称。
        
        返回:
            Skill 名称列表
        """
        return list(self._skills.keys())
    
    def get_all(self) -> Dict[str, BaseSkill]:
        """获取所有已注册的 Skill。
        
        返回:
            Skill 字典（name -> skill）
        """
        return self._skills.copy()
    
    def get_metadata(self) -> List[Dict[str, str]]:
        """获取所有 Skill 的元数据。
        
        返回:
            元数据列表
        """
        return [skill.get_metadata() for skill in self._skills.values()]
