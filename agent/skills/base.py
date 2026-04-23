"""Skill 基类 - 所有 Skill 的抽象基类。

此模块定义了 BaseSkill 抽象类，所有具体的 Skill 都应继承此类。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph

from ..nodes.common import CommonNodes
from ..tools import SQLToolManager
from langchain_core.language_models import BaseChatModel


def _extract_skill_description(md_path: str) -> str:
    """从 SKILL.md 提取路由用描述（## 目的 + ## 适用场景 两段）。"""
    try:
        content = Path(md_path).read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return ""

    lines = content.splitlines()
    sections: Dict[str, list] = {}
    current_section: Optional[str] = None

    for line in lines:
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line)

    parts = []
    for section_name in ("目的", "适用场景"):
        if section_name in sections:
            body = "\n".join(sections[section_name]).strip()
            if body:
                parts.append(f"**{section_name}**\n{body}")

    return "\n\n".join(parts) if parts else ""


class BaseSkill(ABC):
    """Skill 抽象基类。
    
    所有 Skill 都应继承此类并实现 _build_graph 方法。
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tool_manager: SQLToolManager,
        description: str = "",
        skill_md_path: Optional[str] = None,
    ):
        """初始化 Skill。
        
        参数:
            name: Skill 名称
            llm: 语言模型实例
            tool_manager: 工具管理器
            description: Skill 描述（当 skill_md_path 未提供时使用）
            skill_md_path: SKILL.md 文件路径；若提供则自动提取描述，优先于 description
        """
        self.name = name
        self.llm = llm
        self.tool_manager = tool_manager

        if skill_md_path:
            loaded = _extract_skill_description(skill_md_path)
            self.description = loaded if loaded else description
        else:
            self.description = description
        
        # 创建公共节点库（供子类使用）
        self.common = CommonNodes(tool_manager, llm)
        
        # 构建图
        self.graph = self._build_graph()
    
    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """构建 Skill 的子图。
        
        子类必须实现此方法来定义 Skill 的工作流。
        
        返回:
            编译后的 StateGraph
        """
        pass
    
    def invoke(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行 Skill。"""
        if config is None:
            config = {}
        return self.graph.invoke(state, config)
    
    def stream(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """流式执行 Skill。"""
        if config is None:
            config = {}
        return self.graph.stream(state, config)
    
    def get_metadata(self) -> Dict[str, str]:
        """获取 Skill 元数据。"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
        }

