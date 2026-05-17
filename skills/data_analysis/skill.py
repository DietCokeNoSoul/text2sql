"""
Data Analysis Skill Implementation

8-step comprehensive data analysis workflow:
1. understand_goal     → Understand user's analysis objective
2. explore_data        → Explore database structure and statistics
3. plan_analysis       → Generate detailed analysis plan
4. generate_queries    → Create SQL queries for analysis
5. analyze_results     → Analyze query results and extract insights
6. visualize           → Generate visualization recommendations
7. generate_report     → Create comprehensive analysis report
8. export_results      → Export query results to CSV/Excel files
"""

import logging
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START, add_messages

from agent.config import AgentConfig, OutputConfig
from agent.tools import SQLToolManager
from agent.database import SQLDatabaseManager
from agent.skills.base import BaseSkill
from agent.sql_execution_pipeline import run_sql_execution_pipeline
from agent.sql_performance import SQLPerformanceAnalyzer

logger = logging.getLogger(__name__)


def extract_json_from_response(content: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Tries:
    1. Direct JSON parsing
    2. Extract from ```json ... ``` code block
    3. Extract from ``` ... ``` code block
    4. Find first {...} or [...] in text
    """
    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object or array
    brace_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"No valid JSON found in response: {content[:200]}...")


class DataAnalysisSkill(BaseSkill):
    """
    Data Analysis Skill - Comprehensive 7-step data analysis
    
    Flow:
        understand_goal → explore_data → plan_analysis → 
        generate_sql_queries → analyze_results → visualize → generate_report
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tool_manager: SQLToolManager,
        db_manager: SQLDatabaseManager,
        config: Optional[AgentConfig] = None,
        plan_manager: Optional[object] = None,  # SessionPlanManager instance
        confirm_enabled: bool = False,
        sql_correction_max_retries: int = 2,
    ):
        self.db_manager = db_manager
        self._output_config: OutputConfig = config.output if config else OutputConfig()
        self._report_summary_threshold: int = (
            config.memory.report_summary_threshold if config else 800
        )
        sql_perf_cfg = config.sql_perf if config else None
        self._perf_analyzer = SQLPerformanceAnalyzer(
            db_manager=db_manager,
            llm=llm,
            enabled=(sql_perf_cfg.enabled if sql_perf_cfg else True),
            optimize_enabled=(sql_perf_cfg.optimize_enabled if sql_perf_cfg else False),
            rows_warning_threshold=(sql_perf_cfg.rows_warning_threshold if sql_perf_cfg else 10000),
            min_score_improvement=(sql_perf_cfg.min_score_improvement if sql_perf_cfg else 8),
            max_rewrite_rounds=(sql_perf_cfg.max_rewrite_rounds if sql_perf_cfg else 1),
            optimize_score_threshold=(sql_perf_cfg.optimize_score_threshold if sql_perf_cfg else 70),
            semantic_validation_enabled=(sql_perf_cfg.semantic_validation_enabled if sql_perf_cfg else True),
            semantic_sample_rows=(sql_perf_cfg.semantic_sample_rows if sql_perf_cfg else 20),
            optimize_trigger_low_score=(sql_perf_cfg.optimize_trigger_low_score if sql_perf_cfg else True),
            optimize_trigger_large_rows=(sql_perf_cfg.optimize_trigger_large_rows if sql_perf_cfg else True),
            optimize_trigger_full_scan=(sql_perf_cfg.optimize_trigger_full_scan if sql_perf_cfg else True),
            optimize_trigger_filesort=(sql_perf_cfg.optimize_trigger_filesort if sql_perf_cfg else True),
            optimize_trigger_temporary=(sql_perf_cfg.optimize_trigger_temporary if sql_perf_cfg else True),
            optimize_trigger_high_cost=(sql_perf_cfg.optimize_trigger_high_cost if sql_perf_cfg else False),
            optimize_min_triggers=(sql_perf_cfg.optimize_min_triggers if sql_perf_cfg else 1),
            cost_warning_threshold=(sql_perf_cfg.cost_warning_threshold if sql_perf_cfg else 1000.0),
        )
        self._plan_manager = plan_manager
        self.confirm_enabled = confirm_enabled
        self._correction_max_retries = max(0, int(sql_correction_max_retries))
        
        _md = Path(__file__).parent / "SKILL.md"
        super().__init__(
            name="data_analysis",
            llm=llm,
            tool_manager=tool_manager,
            skill_md_path=str(_md),
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the data analysis skill graph"""
        from typing_extensions import TypedDict
        from typing import Annotated
        
        # Define custom state for data analysis
        class DataAnalysisState(TypedDict):
            messages: Annotated[list, add_messages]
            tables: list
            table_schemas: dict  # kept for backward compat (unused after batch refactor)
            combined_schema: str  # all tables' schema in one string (batch fetched)
            analysis_goal: str
            analysis_plan: dict
            sql_queries: list
            query_results: list
            insights: list
            visualizations: list
            chart_files: list   # paths to generated SVG chart files
            export_files: list  # paths to exported CSV/Excel files
            report: str
            task_id: str        # session plan tracking
            constraints: list   # user-defined hard constraints
            thread_id: str      # conversation thread id (injected by _make_skill_node)
        
        graph = StateGraph(DataAnalysisState)
        
        # Add 8 specialized nodes
        graph.add_node("understand_goal", self._understand_goal)
        graph.add_node("explore_data", self._explore_data)
        graph.add_node("plan_analysis", self._plan_analysis)
        graph.add_node("generate_queries", self._generate_queries)
        graph.add_node("analyze_results", self._analyze_results)
        graph.add_node("visualize", self._visualize)
        graph.add_node("generate_report", self._generate_report)
        graph.add_node("export_results", self._export_results_node)
        
        # Build linear flow
        graph.add_edge(START, "understand_goal")
        graph.add_edge("understand_goal", "explore_data")
        graph.add_edge("explore_data", "plan_analysis")
        graph.add_edge("plan_analysis", "generate_queries")
        graph.add_edge("generate_queries", "analyze_results")
        graph.add_edge("analyze_results", "visualize")
        graph.add_edge("visualize", "generate_report")
        graph.add_edge("generate_report", "export_results")
        graph.add_edge("export_results", END)
        
        return graph.compile()

    @staticmethod
    def _latest_human_message(messages: List[Any]) -> str:
        """返回最近一条用户消息内容。"""
        for msg in reversed(messages or []):
            role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
            if role == "human":
                return str(msg.content if hasattr(msg, "content") else msg.get("content", ""))
        return ""
    
    # ============ Node 1: Understand Goal ============
    
    def _understand_goal(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        """
        Understand the user's analysis objective and requirements.
        
        Extract:
        - What questions need to be answered
        - What metrics are important
        - Time ranges or filters
        - Expected output format
        """
        logger.info("[DataAnalysis] Understanding analysis goal")

        messages = state.get("messages", [])
        user_question = self._latest_human_message(messages)
        
        thread_id = state.get("thread_id", "") or (config or {}).get("configurable", {}).get("thread_id", "")
        
        system_prompt = """[Role & Policies]
你是一个数据分析专家，负责理解用户的分析目标并提取结构化需求。
只输出 JSON，不输出 Markdown、解释或前缀内容。

[Task]
分析用户的分析请求，提取结构化的分析目标信息。

[Environment]
（无）

[Evidence]
（无）

[Context]
（无）

[Output]
输出格式（纯 JSON，不含 Markdown）：
{
    "objective": "主要分析目标",
    "metrics": ["指标1", "指标2"],
    "dimensions": ["维度1", "维度2"],
    "filters": {},
    "output_format": "期望输出格式"
}

只返回 JSON 对象，不要任何其他内容。
""" + self._build_constraints_block(state.get("constraints", []))
        
        analysis_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User request: {user_question}\n\nAnalyze this request.")
        ]
        
        response = self.llm.invoke(analysis_messages)
        
        try:
            goal_data = extract_json_from_response(response.content)
            analysis_goal = json.dumps(goal_data, indent=2, ensure_ascii=False)
            
            logger.info(f"[DataAnalysis] Goal understood: {goal_data.get('objective', 'N/A')}")
            
            # ── Session plan: create plan with 7 fixed steps ──────────────
            task_id = ""
            if self._plan_manager:
                task_id = self._plan_manager.new_task_id()
                self._plan_manager.create_plan(
                    task_id=task_id,
                    title=user_question[:80],
                    description=user_question,
                    skill="data_analysis",
                    thread_id=thread_id,
                    steps=[
                        {"step_id": 1, "description": "understand_goal: 理解分析目标", "depends_on": []},
                        {"step_id": 2, "description": "explore_data: 探索数据库结构", "depends_on": [1]},
                        {"step_id": 3, "description": "plan_analysis: 制定分析计划", "depends_on": [2]},
                        {"step_id": 4, "description": "generate_queries: 生成SQL查询", "depends_on": [3]},
                        {"step_id": 5, "description": "analyze_results: 分析结果洞察", "depends_on": [4]},
                        {"step_id": 6, "description": "visualize: 生成可视化建议", "depends_on": [5]},
                        {"step_id": 7, "description": "generate_report: 生成分析报告", "depends_on": [6]},
                        {"step_id": 8, "description": "export_results: 导出CSV/Excel结果文件", "depends_on": [7]},
                    ],
                )
                self._plan_manager.update_step(task_id, 1, "done",
                    result_summary=goal_data.get("objective", ""))
                logger.info(f"[SessionPlan] Created plan {task_id}")
            
            new_message = AIMessage(
                content=f"Analysis Goal:\n{analysis_goal}"
            )
            
            return {
                "messages": [new_message],
                "analysis_goal": analysis_goal,
                "task_id": task_id,
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[DataAnalysis] Failed to parse goal: {e}")
            error_message = AIMessage(content=f"Failed to understand goal: {str(e)}")
            return {"messages": [error_message], "analysis_goal": "{}"}
    
    # ── Session plan helpers ───────────────────────────────────────────────

    def _step_start(self, state: Dict[str, Any], step_id: int) -> None:
        """Mark a step as in_progress in the session plan."""
        task_id = state.get("task_id", "")
        if self._plan_manager and task_id:
            self._plan_manager.update_step(task_id, step_id, "in_progress")

    def _step_done(self, state: Dict[str, Any], step_id: int, summary: str = "") -> None:
        """Mark a step as done in the session plan."""
        task_id = state.get("task_id", "")
        if self._plan_manager and task_id:
            self._plan_manager.update_step(task_id, step_id, "done", result_summary=summary[:200])

    # ============ Node 2: Explore Data ============
    
    def _explore_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explore database structure and gather statistics.
        
        Collect:
        - Available tables
        - Table schemas
        - Row counts
        - Key relationships
        """
        logger.info("[DataAnalysis] Exploring data")
        self._step_start(state, 2)
        
        # List tables (uses cache if available)
        list_tables_tool = self.tool_manager.get_list_tables_tool()
        tables_result = list_tables_tool.invoke({})
        tables = [t.strip() for t in tables_result.split(',') if t.strip()]
        
        # Batch fetch schema for ALL tables in one call (cache-friendly, same key as Simple/Complex Skills)
        combined_schema = ""
        try:
            get_schema_tool = self.tool_manager.get_schema_tool()
            table_names_str = ", ".join(tables)
            combined_schema = get_schema_tool.invoke({"table_names": table_names_str})
            logger.info(f"[DataAnalysis] Batch schema fetched: {len(tables)} tables, {len(combined_schema)} chars")
        except Exception as e:
            logger.warning(f"[DataAnalysis] Failed to batch fetch schema: {e}")
        
        exploration_summary = f"Found {len(tables)} tables: {', '.join(tables)}. Schema loaded ({len(combined_schema)} chars)."
        new_message = AIMessage(content=exploration_summary)
        self._step_done(state, 2, exploration_summary)
        
        return {
            "messages": [new_message],
            "tables": tables,
            "combined_schema": combined_schema,
            "table_schemas": {}  # kept for state compat, logic now uses combined_schema
        }
    
    # ============ Node 3: Plan Analysis ============
    
    def _plan_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed analysis plan based on goal and data.
        
        Create:
        - Step-by-step analysis approach
        - Required calculations
        - Expected insights
        """
        logger.info("[DataAnalysis] Planning analysis")
        self._step_start(state, 3)
        
        analysis_goal = state.get("analysis_goal", "{}")
        combined_schema = state.get("combined_schema", "")
        
        # Provide a truncated schema summary for planning context
        data_summary = combined_schema[:3000] + "..." if len(combined_schema) > 3000 else combined_schema
        
        system_prompt = f"""[Role & Policies]
你是一个数据分析规划专家，基于目标和数据结构制定详细的分析计划。
只输出 JSON，不添加任何解释或 Markdown 包装。

[Task]
根据分析目标和可用数据，生成分步骤的分析计划。

[Environment]
（无）

[Evidence]
分析目标：
{analysis_goal}

可用数据概要：
{data_summary}

[Context]
（无）

[Output]
输出格式（纯 JSON）：
{{
    "steps": [
        {{"id": 1, "description": "...", "tables": ["..."], "calculations": ["..."]}},
        {{"id": 2, "description": "...", "tables": ["..."], "calculations": ["..."]}}
    ],
    "expected_insights": ["...", "..."],
    "visualization_suggestions": ["...", "..."]
}}

只返回 JSON 对象，不要任何其他内容。{self._build_constraints_block(state.get("constraints", []))}
"""
        
        plan_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate the analysis plan.")
        ]
        
        response = self.llm.invoke(plan_messages)
        
        try:
            plan_data = extract_json_from_response(response.content)
            
            logger.info(f"[DataAnalysis] Plan created with {len(plan_data.get('steps', []))} steps")
            self._step_done(state, 3, f"{len(plan_data.get('steps', []))} analysis steps planned")
            
            new_message = AIMessage(
                content=f"Analysis Plan:\n{json.dumps(plan_data, indent=2, ensure_ascii=False)}"
            )
            
            return {
                "messages": [new_message],
                "analysis_plan": plan_data
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[DataAnalysis] Failed to parse plan: {e}")
            # Return empty plan to allow workflow to continue
            empty_plan = {"steps": [], "expected_insights": [], "visualization_suggestions": []}
            error_message = AIMessage(content=f"Failed to create plan: {str(e)}")
            return {"messages": [error_message], "analysis_plan": empty_plan}
    
    # ============ Node 4: Generate SQL Queries ============
    
    def _generate_queries(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SQL queries based on the analysis plan.
        """
        logger.info("[DataAnalysis] Generating SQL queries")
        self._step_start(state, 4)
        
        analysis_plan = state.get("analysis_plan", {})
        analysis_goal = state.get("analysis_goal", "{}")
        combined_schema = state.get("combined_schema", "")
        
        steps = analysis_plan.get("steps", [])
        
        # If no plan steps, generate queries directly from goal
        if not steps:
            logger.warning("[DataAnalysis] No plan steps, generating queries from goal directly")
            return self._generate_queries_from_goal(state)
        
        sql_queries = []
        
        # Use the pre-fetched combined schema directly
        all_schemas = combined_schema
        
        for step in steps:
            step_id = step.get("id")
            description = step.get("description", "")
            calculations = step.get("calculations", [])
            
            system_prompt = f"""[Role & Policies]
你是一个 MySQL SQL 生成专家，为数据分析步骤生成可执行的 SQL 查询。
只输出纯 SQL，不加任何解释或 Markdown 包装。

[Task]
为以下分析步骤生成 MySQL 兼容的 SQL 查询。
步骤：{description}
所需计算：{', '.join(calculations) if calculations else '按需'}

[Environment]
- 数据库方言：MySQL

[Evidence]
可用表和 Schema：
{all_schemas}

[Context]
（无）

[Output]
只输出 SQL 查询语句，不加 Markdown、不加解释。{self._build_constraints_block(state.get("constraints", []))}
"""
            
            query_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate the SQL query.")
            ]
            
            response = self.llm.invoke(query_messages)
            sql_query = response.content.strip()
            
            # Remove markdown code blocks if present
            if sql_query.startswith("```"):
                lines = sql_query.split("\n")
                sql_query = "\n".join(lines[1:-1]).strip()
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3].strip()
            # Remove leading "sql" if present
            if sql_query.lower().startswith("sql\n"):
                sql_query = sql_query[4:].strip()

            # ── 用户禁令守卫 ──────────────────────────────────────────────────
            from agent.constraint_guard import check_constraints, build_block_message
            from agent.sql_errors import UserPolicyError
            try:
                check_constraints(sql_query, state.get("constraints", []))
            except UserPolicyError as ce:
                logger.warning("[DataAnalysis] Blocked by constraint: %r", ce.matched_keyword)
                sql_queries.append({
                    "step_id": step_id,
                    "description": description,
                    "query": None,
                    "blocked": True,
                    "block_message": build_block_message(ce),
                })
                continue

            sql_queries.append({
                "step_id": step_id,
                "description": description,
                "query": sql_query
            })
            
            logger.info(f"[DataAnalysis] Generated SQL for step {step_id}: {sql_query[:80]}")
        
        new_message = AIMessage(
            content=f"Generated {len(sql_queries)} SQL queries for analysis."
        )
        self._step_done(state, 4, f"{len(sql_queries)} queries generated")

        # ── Session plan: add per-SQL sub-steps for granular tracking ─────
        task_id = state.get("task_id", "")
        if self._plan_manager and task_id:
            for sq in sql_queries:
                if not sq.get("blocked"):
                    self._plan_manager.add_step(task_id, {
                        "step_id": 40 + sq["step_id"],
                        "description": f"SQL-{sq['step_id']}: {sq['description']}",
                        "query": sq.get("query", ""),
                        "status": "pending",
                    })
        
        return {
            "messages": [new_message],
            "sql_queries": sql_queries
        }
    
    def _generate_queries_from_goal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: generate queries directly from goal when plan parsing fails."""
        analysis_goal = state.get("analysis_goal", "{}")
        combined_schema = state.get("combined_schema", "")
        
        all_schemas = combined_schema
        
        system_prompt = f"""[Role & Policies]
你是一个数据分析 SQL 生成专家。只输出 JSON，不输出 Markdown 或解释。

[Task]
根据分析目标，直接生成覆盖所有分析需求的 SQL 查询列表。

[Environment]
- 数据库方言：MySQL

[Evidence]
分析目标：
{analysis_goal}

可用表和 Schema：
{all_schemas}

[Context]
（无）

[Output]
输出格式（纯 JSON 数组）：
[
    {{"step_id": 1, "description": "...", "query": "SELECT ..."}},
    {{"step_id": 2, "description": "...", "query": "SELECT ..."}}
]
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate SQL queries as JSON array.")
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            sql_queries = extract_json_from_response(response.content)
            if isinstance(sql_queries, dict):
                sql_queries = sql_queries.get("queries", [])
            logger.info(f"[DataAnalysis] Generated {len(sql_queries)} queries from goal directly")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[DataAnalysis] Failed to generate queries from goal: {e}")
            sql_queries = []
        
        return {
            "messages": [AIMessage(content=f"Generated {len(sql_queries)} queries from goal.")],
            "sql_queries": sql_queries
        }
    
    # ============ Node 5: Analyze Results ============
    
    def _analyze_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute queries and analyze results to extract insights.
        
        Run SQL queries and interpret the results.
        """
        logger.info("[DataAnalysis] Analyzing results")
        self._step_start(state, 5)
        
        sql_queries = state.get("sql_queries", [])
        query_tool = self.tool_manager.get_query_tool()
        
        query_results = []
        insights = []
        
        for query_data in sql_queries:
            step_id = query_data.get("step_id")
            description = query_data.get("description")
            sql = query_data.get("query")

            # 被禁令拦截的步骤：跳过执行，记录拦截结果
            if query_data.get("blocked"):
                query_results.append({
                    "step_id": step_id,
                    "description": description,
                    "success": False,
                    "result": query_data.get("block_message", "该步骤已被禁令拦截"),
                    "error": "blocked_by_constraint",
                })
                continue

            perf_opt_result = self._perf_analyzer.optimize_sql(
                sql,
                constraints=state.get("constraints", []),
            )
            sql = perf_opt_result.final_sql
            perf_analysis = perf_opt_result.final_analysis
            logger.info("[DataAnalysis][SQLPerf][step=%s][before] %s", step_id, perf_opt_result.original_analysis.summary)
            logger.info("[DataAnalysis][SQLPerf][step=%s][after] %s", step_id, perf_analysis.summary)
            if perf_opt_result.optimized:
                logger.info("[DataAnalysis][SQLPerf][step=%s] SQL optimized and replaced", step_id)

            if self.confirm_enabled and sql:
                from agent.sql_confirm import prompt_sql_confirmation, build_skip_message
                action, reason = prompt_sql_confirmation(sql)
                if action == "skip":
                    skip_content = build_skip_message(sql, reason)
                    logger.info(f"[DataAnalysis] Step {step_id} skipped by user")
                    query_results.append({
                        "step_id": step_id,
                        "description": description,
                        "query": sql,
                        "original_query": sql,
                        "result": skip_content,
                        "success": False,
                        "skipped": True,
                        "retries": 0,
                        "performance": perf_analysis.to_dict(),
                        "performance_optimization": perf_opt_result.to_dict(),
                    })
                    insights.append({"step_id": step_id, "insight": "SQL已被用户跳过"})
                    continue

            pipeline_result = run_sql_execution_pipeline(
                sql=sql,
                query_tool=query_tool,
                db_manager=self.db_manager,
                llm=self.llm,
                perf_analyzer=self._perf_analyzer,
                constraints=state.get("constraints", []),
                context_label=f"[DataAnalysis] step {step_id}",
                correction_max_retries=self._correction_max_retries,
                precomputed_optimization=perf_opt_result,
            )
            logger.info("[DataAnalysis][Pipeline][step=%s] decision=%s", step_id, pipeline_result.decision)

            if pipeline_result.success:
                result = pipeline_result.result
                query_results.append({
                    "step_id": step_id,
                    "description": description,
                    "query": pipeline_result.final_sql,
                    "original_query": sql,
                    "result": result,
                    "success": True,
                    "retries": pipeline_result.retries,
                    "correction_trace": pipeline_result.correction_trace,
                    "pipeline_decision": pipeline_result.decision,
                    "performance": perf_analysis.to_dict(),
                    "performance_optimization": perf_opt_result.to_dict(),
                })

                # ── Emit SQL step event for live frontend ─────────────────
                from agent import sql_step_emitter
                sql_step_emitter.emit(
                    str(step_id),
                    description,
                    pipeline_result.final_sql,
                    performance=perf_analysis.to_dict(),
                    optimization=perf_opt_result.to_dict(),
                )

                # ── Session plan: mark sub-step done ──────────────────────
                task_id = state.get("task_id", "")
                if self._plan_manager and task_id:
                    self._plan_manager.update_step(
                        task_id, 40 + step_id, "done",
                        sql=pipeline_result.final_sql,
                        result_summary=str(result)[:200],
                        notes=(
                            perf_analysis.summary + (
                                f" | 优化: {perf_opt_result.original_analysis.score}->{perf_analysis.score}"
                                if perf_opt_result.optimized else ""
                            )
                        )[:200],
                    )

                # Generate insight from result
                insight_prompt = f"""[Role & Policies]
你是一个数据洞察提取专家，基于 SQL 查询结果提炼关键业务发现。

[Task]
从以下查询结果中提取关键洞察，用简洁中文描述（2-4句话）。

[Environment]
查询目的：{description}

[Evidence]
查询结果：{result[:500]}

[Context]
（无）

[Output]
用简洁中文描述关键发现，2-4句话，直接输出文字，不加前缀。
"""
                insight_messages = [
                    SystemMessage(content="你是一个数据洞察提取专家。"),
                    HumanMessage(content=insight_prompt),
                ]
                insight_response = self.llm.invoke(insight_messages)
                insights.append({
                    "step_id": step_id,
                    "insight": insight_response.content,
                })
                logger.info(f"[DataAnalysis] Analyzed step {step_id} (retries={pipeline_result.retries})")
            else:
                error_str = pipeline_result.error
                logger.error(f"[DataAnalysis] Query failed for step {step_id} after correction: {error_str}")
                query_results.append({
                    "step_id": step_id,
                    "description": description,
                    "query": pipeline_result.final_sql,
                    "error": error_str,
                    "success": False,
                    "retries": pipeline_result.retries,
                    "correction_trace": pipeline_result.correction_trace,
                    "pipeline_decision": pipeline_result.decision,
                    "performance": perf_analysis.to_dict(),
                    "performance_optimization": perf_opt_result.to_dict(),
                })
        
        new_message = AIMessage(
            content=f"Executed {len(query_results)} queries, extracted {len(insights)} insights."
        )
        self._step_done(state, 5, f"{len(insights)} insights from {len(query_results)} queries")

        # Add SQL step messages for history reconstruction
        sql_msgs = [new_message]
        for qr in query_results:
            if qr.get("success") and qr.get("query"):
                sql_msgs.append(AIMessage(
                    content=f"__sql__:{qr['step_id']}:{qr['description']}:{qr['query']}"
                ))
        
        return {
            "messages": sql_msgs,
            "query_results": query_results,
            "insights": insights
        }
    
    # ============ Node 6: Visualize ============
    
    def _visualize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization recommendations AND actual SVG chart files.
        """
        logger.info("[DataAnalysis] Generating visualizations")
        self._step_start(state, 6)
        
        from skills.data_analysis.chart_generator import ChartGenerator

        query_results = state.get("query_results", [])
        insights = state.get("insights", [])

        # Resolve chart output directory from config (relative paths anchored to project root)
        chart_dir = self._output_config.chart_dir
        if not os.path.isabs(chart_dir):
            skill_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(skill_file)))
            output_dir = os.path.join(project_root, chart_dir)
        else:
            output_dir = chart_dir
        
        visualizations = []
        chart_files = []

        for result in query_results:
            if not result.get("success"):
                continue

            step_id = result.get("step_id")
            description = result.get("description", "")
            data = result.get("result", "")

            # ── Ask LLM for chart recommendation ──────────────────────────
            viz_prompt = f"""[Role & Policies]
你是一个数据可视化专家，为查询结果推荐最合适的图表类型。只输出 JSON，不添加解释。

[Task]
为以下查询结果推荐可视化方案。

[Environment]
- 支持的图表类型：bar（柱状图）、pie（饼图）、line（折线图）

[Evidence]
数据描述：{description}
样本数据：{str(data)[:300]}

[Context]
（无）

[Output]
输出格式（纯 JSON）：
{{
    "chart_type": "bar",
    "x_axis": "X轴描述",
    "y_axis": "Y轴描述",
    "title": "中文图表标题",
    "message": "可视化说明"
}}"""

            viz_messages = [
                SystemMessage(content="你是一个数据可视化专家，只输出 JSON。"),
                HumanMessage(content=viz_prompt)
            ]

            response = self.llm.invoke(viz_messages)

            try:
                viz_data = extract_json_from_response(response.content)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"[DataAnalysis] Failed to parse viz for step {step_id}")
                viz_data = {"chart_type": "bar", "x_axis": "", "y_axis": "", "title": description, "message": ""}

            chart_type = viz_data.get("chart_type", "bar").lower()
            if chart_type not in ("bar", "pie", "line"):
                chart_type = "bar"
            title = viz_data.get("title") or description or f"Step {step_id}"

            visualizations.append({"step_id": step_id, **viz_data})

            # ── Actually generate the SVG chart ───────────────────────────
            chart_path = ChartGenerator.from_query_result(
                raw_result=str(data),
                chart_type=chart_type,
                title=title,
                x_label=viz_data.get("x_axis", ""),
                y_label=viz_data.get("y_axis", ""),
                output_dir=output_dir,
            )
            if chart_path:
                chart_files.append(chart_path)
                logger.info(f"[DataAnalysis] Chart generated: {chart_path}")

        chart_summary = (
            f"生成了 {len(chart_files)} 个图表文件到 {output_dir}"
            if chart_files else "查询结果无法转换为图表（数据格式不支持）"
        )
        new_message = AIMessage(
            content=f"Generated {len(visualizations)} viz recommendations. {chart_summary}."
        )
        self._step_done(state, 6, f"{len(visualizations)} visualizations, {len(chart_files)} charts")

        return {
            "messages": [new_message],
            "visualizations": visualizations,
            "chart_files": chart_files,
        }
    
    # ============ Node 7: Generate Report ============
    
    def _generate_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report using LLM.
        Appends generated chart file paths at the end of the report.
        """
        logger.info("[DataAnalysis] Generating analysis report")
        self._step_start(state, 7)
        
        analysis_goal = state.get("analysis_goal", "{}")
        insights = state.get("insights", [])
        visualizations = state.get("visualizations", [])
        query_results = state.get("query_results", [])
        chart_files = state.get("chart_files", [])
        
        # Build context for report generation
        insights_text = "\n".join([
            f"- {item.get('insight', '')}" for item in insights
        ])
        
        results_text = ""
        for r in query_results:
            status = "成功" if r.get("success") else "失败"
            desc = r.get("description", "")
            data = str(r.get("result", r.get("error", "")))[:200]
            results_text += f"[{status}] {desc}: {data}\n"
        
        viz_text = "\n".join([
            f"- {v.get('chart_type', 'N/A')}: {v.get('title', 'N/A')} - {v.get('message', '')}"
            for v in visualizations
        ])
        
        system_prompt = f"""[Role & Policies]
你是一个数据分析报告撰写专家，生成结构清晰、有数据支撑的中文分析报告。
直接输出 Markdown，不包裹在代码块中，不添加解释前缀。

[Task]
根据提供的分析结果、洞察和可视化建议，生成完整的数据分析报告。

[Environment]
（无）

[Evidence]
分析目标：
{analysis_goal}

查询结果：
{results_text}

关键洞察：
{insights_text}

可视化建议：
{viz_text}

[Context]
（无）

[Output]
生成结构化 Markdown 报告，包含：
1. 报告标题和摘要
2. 关键发现（用数据支撑）
3. 可视化建议
4. 结论和建议

直接输出 Markdown，不要包裹在代码块中。
"""
        
        report_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="请生成分析报告。")
        ]
        
        response = self.llm.invoke(report_messages)
        report_content = response.content.strip()
        
        # Remove markdown code block wrapper if present
        if report_content.startswith("```"):
            lines = report_content.split("\n")
            report_content = "\n".join(lines[1:])
        if report_content.endswith("```"):
            report_content = report_content[:-3].strip()
        
        logger.info(f"[DataAnalysis] Report generated ({len(report_content)} chars)")
        
        # Append chart file references at end of report
        if chart_files:
            charts_section = "\n\n---\n\n## 📊 生成的图表\n\n"
            for path in chart_files:
                title = os.path.splitext(os.path.basename(path))[0]  # e.g. bar_店铺总数
                # Use relative path from the report file's directory
                rel_path = os.path.join("charts", os.path.basename(path)).replace("\\", "/")
                charts_section += f"![{title}]({rel_path})\n\n"
            report_content += charts_section
        
        # ── Save report to project's report/ folder ────────────────────
        report_path = self._save_report(report_content)
        if report_path:
            report_content += f"\n\n---\n\n> 报告已保存至: `{report_path}`\n"
            logger.info(f"[DataAnalysis] Report saved: {report_path}")

        # ── 即时摘要：报告超过阈值时生成结构化摘要写入消息历史 ────────────────
        # UI 展示完整报告，消息历史存储压缩摘要，避免长报告占满上下文窗口
        if len(report_content) > self._report_summary_threshold:
            try:
                compress_prompt = f"""[Role & Policies]
你是报告摘要专家，用结构化格式压缩分析报告，保留所有关键数字和结论。

[Task]
将以下分析报告压缩为 200 字以内的结构化摘要。

[Environment]
（无）

[Evidence]
{report_content[:3000]}

[Context]
（无）

[Output]
严格使用以下格式输出，不加其他内容：
- 目标：（一句话描述分析目的）
- 关键指标：（数值型结果，如 总记录数: 7, 最高销量: 2340）
- 核心结论：（1-2句话）
- 最优项：（表现最好的维度/值）
- 异常项：（如有异常则填写，否则写"无"）
"""
                compress_messages = [
                    SystemMessage(content=compress_prompt),
                    HumanMessage(content="请压缩报告。"),
                ]
                compressed = self.llm.invoke(compress_messages).content.strip()
                history_content = f"[报告摘要]\n{compressed}"
                logger.info(
                    f"[DataAnalysis] Report compressed: {len(report_content)} → {len(history_content)} chars"
                )
            except Exception as e:
                logger.warning(f"[DataAnalysis] Report compression failed: {e}, using truncation")
                history_content = report_content[:self._report_summary_threshold] + "…[已截断，完整报告已保存]"
        else:
            history_content = report_content

        # 消息历史写入摘要版，UI 展示完整版（通过 state["report"] 传递）
        new_message = AIMessage(content=history_content)
        self._step_done(state, 7, f"Report generated ({len(report_content)} chars)")
        
        return {
            "messages": [new_message],
            "report": report_content  # UI 读取此字段展示完整报告
        }
    
    def _export_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node 8: Export query results to CSV (and optionally Excel) files.

        Each successful query result is written to its own CSV file in the
        configured report directory.  If openpyxl is installed, an aggregated
        Excel workbook (.xlsx) is also produced with one sheet per query.
        """
        logger.info("[DataAnalysis] Exporting query results")
        self._step_start(state, 8)

        query_results = state.get("query_results", [])
        task_id = state.get("task_id", "")

        # Resolve export directory (same as report dir)
        report_dir = self._output_config.report_dir
        if not os.path.isabs(report_dir):
            skill_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(skill_file)))
            report_dir = os.path.join(project_root, report_dir)

        os.makedirs(report_dir, exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"export_{task_id or timestamp}"

        export_files: list[str] = []

        # ── per-query CSV ──────────────────────────────────────────────────
        for idx, qr in enumerate(query_results, start=1):
            if not qr.get("success"):
                continue
            raw = qr.get("result", "")
            rows = self._parse_query_result(raw)
            if not rows:
                continue

            csv_path = os.path.join(report_dir, f"{prefix}_step{idx}.csv")
            try:
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    for row in rows:
                        writer.writerow(row)
                export_files.append(csv_path)
                logger.info(f"[DataAnalysis] CSV exported: {csv_path}")
            except Exception as e:
                logger.warning(f"[DataAnalysis] Failed to write CSV step{idx}: {e}")

        # ── aggregated Excel (optional, requires openpyxl) ─────────────────
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            wb.remove(wb.active)  # remove default empty sheet
            written = 0
            for idx, qr in enumerate(query_results, start=1):
                if not qr.get("success"):
                    continue
                raw = qr.get("result", "")
                rows = self._parse_query_result(raw)
                if not rows:
                    continue
                sheet_name = f"Step{idx}"
                ws = wb.create_sheet(title=sheet_name)
                for row in rows:
                    ws.append(list(row))
                written += 1
            if written:
                xlsx_path = os.path.join(report_dir, f"{prefix}.xlsx")
                wb.save(xlsx_path)
                export_files.append(xlsx_path)
                logger.info(f"[DataAnalysis] Excel exported: {xlsx_path}")
        except ImportError:
            logger.debug("[DataAnalysis] openpyxl not installed; skipping Excel export")
        except Exception as e:
            logger.warning(f"[DataAnalysis] Failed to write Excel: {e}")

        summary = f"导出了 {len(export_files)} 个文件到 {report_dir}" if export_files else "无可导出数据"
        self._step_done(state, 8, summary)
        
        # ── Session plan: mark overall task complete ───────────────────────
        task_id = state.get("task_id", "")
        if self._plan_manager and task_id:
            self._plan_manager.mark_complete(task_id, success=True)
            logger.info(f"[SessionPlan] Task {task_id} complete")
        
        return {
            "messages": [AIMessage(content=summary)],
            "export_files": export_files,
        }

    def _parse_query_result(self, raw: str) -> list[list]:
        """
        Parse a raw SQL query result string into a list of rows (each row is a list).

        Handles two common formats:
        - LangChain SQLDatabase tuple strings: [(a, b), (c, d)]
        - Tab/comma-separated plain text
        """
        if not raw or not raw.strip():
            return []

        stripped = raw.strip()

        # Try as Python literal (list of tuples)
        try:
            import ast
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list) and parsed:
                return [list(row) if isinstance(row, (tuple, list)) else [row] for row in parsed]
        except Exception:
            pass

        # Fall back to newline-separated rows
        lines = [l for l in stripped.splitlines() if l.strip()]
        if not lines:
            return []
        # Detect separator: tab first, then comma
        sep = "\t" if "\t" in lines[0] else ","
        return [line.split(sep) for line in lines]

    def _save_report(self, content: str) -> Optional[str]:
        """保存报告 Markdown 文件到配置的 report 目录，文件名含时间戳。"""
        import datetime
        
        # Resolve report directory from config (relative paths anchored to project root)
        report_dir = self._output_config.report_dir
        if not os.path.isabs(report_dir):
            skill_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(skill_file)))
            report_dir = os.path.join(project_root, report_dir)
        
        try:
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.md"
            filepath = os.path.join(report_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return filepath
        except Exception as e:
            logger.error(f"[DataAnalysis] Failed to save report: {e}")
            return None
