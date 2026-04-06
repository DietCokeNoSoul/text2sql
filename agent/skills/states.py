"""Skill 状态定义 - 各个 Skill 使用的 State 类。

此模块定义了所有 Skill 使用的状态类，包括字段和类型注解。
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.messages import AnyMessage


# ============ Simple Query Skill State ============

class SimpleQueryState(BaseModel):
    """简单查询 Skill 的状态。
    
    用于处理单表或简单多表查询。
    """
    
    # 消息历史
    messages: List[AnyMessage] = Field(default_factory=list)
    
    # 数据库信息
    tables: List[str] = Field(default_factory=list)
    table_schema: str = ""  # 重命名避免与 BaseModel.schema() 冲突
    
    # 查询
    sql: str = ""
    query_result: Optional[Any] = None


# ============ Complex Query Skill State ============

class ComplexQueryState(BaseModel):
    """复杂查询 Skill 的状态。
    
    使用 Plan-Execute 模式处理复杂查询。
    """
    
    # 消息历史
    messages: List[AnyMessage] = Field(default_factory=list)
    
    # 数据库信息
    tables: List[str] = Field(default_factory=list)
    table_schema: str = ""  # 重命名避免与 BaseModel.schema() 冲突
    
    # 查询计划
    query_plan: Dict[str, Any] = Field(default_factory=dict)
    # 格式: {
    #     "steps": [
    #         {"id": "step1", "description": "...", "status": "pending"},
    #         {"id": "step2", "description": "...", "status": "pending"}
    #     ],
    #     "dependencies": {...}
    # }
    
    # 子查询结果（每个 step 的结果）
    step_results: Dict[str, Any] = Field(default_factory=dict)
    # 格式: {"step1": {"sql": "...", "result": ...}, "step2": ...}
    
    # 最终聚合结果
    final_result: Optional[str] = None
    
    # 计划完成状态
    plan_completed: bool = False


# ============ Data Analysis Skill State ============

class DataAnalysisState(BaseModel):
    """数据分析 Skill 的状态。
    
    完整的数据分析流程：探索 → 规划 → 分析 → 报告。
    """
    
    # 输入
    messages: List[AnyMessage] = Field(default_factory=list)
    analysis_goal: str = ""  # 分析目标
    
    # 数据库信息 (由公共节点填充)
    tables: List[str] = Field(default_factory=list)
    table_schemas: Dict[str, str] = Field(default_factory=dict)  # 表名 -> DDL，重命名避免冲突
    
    # 数据探索结果
    data_samples: Dict[str, List[Dict]] = Field(default_factory=dict)  # 表名 -> 样本数据
    statistics: Dict[str, Dict] = Field(default_factory=dict)  # 表名 -> 统计信息
    
    # 分析计划
    analysis_plan: Dict[str, Any] = Field(default_factory=dict)
    # 格式: {
    #     "steps": [
    #         {"step": 1, "purpose": "...", "query_type": "...", "expected_insight": "..."}
    #     ],
    #     "key_metrics": ["metric1", "metric2"]
    # }
    
    # 查询和结果
    sql_queries: List[Dict[str, str]] = Field(default_factory=list)  
    # 格式: [{"purpose": "...", "sql": "...", "expected_insight": "..."}]
    
    query_results: List[Dict[str, Any]] = Field(default_factory=list)
    # 格式: [{"purpose": "...", "sql": "...", "data": ..., "status": "success/failed"}]
    
    # 分析结果
    insights: List[Dict[str, str]] = Field(default_factory=list)  
    # 格式: [{"query_purpose": "...", "insight": "..."}]
    
    visualization_suggestions: List[Dict[str, str]] = Field(default_factory=list)
    # 格式: [{"chart_type": "bar", "x_axis": "...", "y_axis": "...", "title": "...", "description": "..."}]
    
    # 最终输出
    report: str = ""  # 分析报告


# ============ Main Graph State ============

class MainGraphState(BaseModel):
    """主图的状态。
    
    用于在不同 Skill 之间路由和协调。
    """
    
    # 用户输入
    messages: List[AnyMessage] = Field(default_factory=list)
    
    # 路由决策
    query_type: str = ""  # "simple" | "complex" | "analysis"
    
    # Skill 执行结果
    skill_result: Optional[Any] = None
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict)
