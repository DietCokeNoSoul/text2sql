"""Skill 基类 - 所有 Skill 的抽象基类。

此模块定义了 BaseSkill 抽象类，所有具体的 Skill 都应继承此类。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from langgraph.graph import StateGraph

from ..nodes.common import CommonNodes
from ..tools import SQLToolManager
from langchain.chat_models import BaseChatModel


class BaseSkill(ABC):
    """Skill 抽象基类。
    
    所有 Skill 都应继承此类并实现 _build_graph 方法。
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tool_manager: SQLToolManager,
        description: str = ""
    ):
        """初始化 Skill。
        
        参数:
            name: Skill 名称
            llm: 语言模型实例
            tool_manager: 工具管理器
            description: Skill 描述
        """
        self.name = name
        self.llm = llm
        self.tool_manager = tool_manager
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
        """执行 Skill。
        
        参数:
            state: 输入状态
            config: 配置（可选）
            
        返回:
            输出状态
        """
        if config is None:
            config = {}
        
        return self.graph.invoke(state, config)
    
    def stream(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """流式执行 Skill。
        
        参数:
            state: 输入状态
            config: 配置（可选）
            
        返回:
            流式迭代器
        """
        if config is None:
            config = {}
        
        return self.graph.stream(state, config)
    
    def get_metadata(self) -> Dict[str, str]:
        """获取 Skill 元数据。
        
        返回:
            包含名称、描述等信息的字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }
