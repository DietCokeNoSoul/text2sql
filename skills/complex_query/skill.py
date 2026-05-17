"""
Complex Query Skill Implementation

Plan-Execute pattern:
1. list_tables → get_schema (reuse common nodes)
2. dual_tower_retrieve → semantic column search + Steiner Tree JOIN planning (optional)
3. plan → analyze question, generate multi-step query plan
4. execute_step → parallel execution using Send API
5. aggregate → collect all step results
6. judge → check if all steps completed, loop if needed
"""

import logging
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.types import Send

from agent.tools import SQLToolManager
from agent.database import SQLDatabaseManager
from agent.skills.base import BaseSkill
from agent.skills.states import ComplexQueryState
from agent.sql_execution_pipeline import run_sql_execution_pipeline
from agent.sql_performance import SQLPerformanceAnalyzer

logger = logging.getLogger(__name__)


class ComplexQuerySkill(BaseSkill):
    """
    Complex Query Skill - Plan-Execute pattern with parallel execution
    
    Flow:
        list_tables → get_schema → [dual_tower_retrieve] → plan
        → execute_steps (parallel) → aggregate → judge
        
    Uses Send API for parallel execution of sub-queries.
    Optionally uses DualTowerRetriever for schema pruning before planning.
    Optionally uses SessionPlanManager for session-level task tracking.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tool_manager: SQLToolManager,
        db_manager: SQLDatabaseManager,
        retriever: Optional[object] = None,   # DualTowerRetriever instance
        plan_manager: Optional[object] = None, # SessionPlanManager instance
        confirm_enabled: bool = False,
        sql_correction_max_retries: int = 2,
        sql_perf_enabled: bool = True,
        sql_perf_rows_threshold: int = 10000,
        sql_perf_optimize_enabled: bool = False,
        sql_perf_max_rewrite_rounds: int = 1,
        sql_perf_min_improvement: int = 8,
        sql_perf_optimize_threshold: int = 70,
        sql_perf_semantic_validation_enabled: bool = True,
        sql_perf_semantic_sample_rows: int = 20,
        sql_perf_optimize_trigger_low_score: bool = True,
        sql_perf_optimize_trigger_large_rows: bool = True,
        sql_perf_optimize_trigger_full_scan: bool = True,
        sql_perf_optimize_trigger_filesort: bool = True,
        sql_perf_optimize_trigger_temporary: bool = True,
        sql_perf_optimize_trigger_high_cost: bool = False,
        sql_perf_optimize_min_triggers: int = 1,
        sql_perf_cost_warning_threshold: float = 1000.0,
    ):
        self.db_manager = db_manager
        self._retriever = retriever
        self._plan_manager = plan_manager
        self.confirm_enabled = confirm_enabled
        self._correction_max_retries = max(0, int(sql_correction_max_retries))
        self._perf_analyzer = SQLPerformanceAnalyzer(
            db_manager=db_manager,
            llm=llm,
            enabled=sql_perf_enabled,
            optimize_enabled=sql_perf_optimize_enabled,
            rows_warning_threshold=sql_perf_rows_threshold,
            min_score_improvement=sql_perf_min_improvement,
            max_rewrite_rounds=sql_perf_max_rewrite_rounds,
            optimize_score_threshold=sql_perf_optimize_threshold,
            semantic_validation_enabled=sql_perf_semantic_validation_enabled,
            semantic_sample_rows=sql_perf_semantic_sample_rows,
            optimize_trigger_low_score=sql_perf_optimize_trigger_low_score,
            optimize_trigger_large_rows=sql_perf_optimize_trigger_large_rows,
            optimize_trigger_full_scan=sql_perf_optimize_trigger_full_scan,
            optimize_trigger_filesort=sql_perf_optimize_trigger_filesort,
            optimize_trigger_temporary=sql_perf_optimize_trigger_temporary,
            optimize_trigger_high_cost=sql_perf_optimize_trigger_high_cost,
            optimize_min_triggers=sql_perf_optimize_min_triggers,
            cost_warning_threshold=sql_perf_cost_warning_threshold,
        )
        self.simple_query_tool = None  # Will be set by main graph if needed
        
        _md = Path(__file__).parent / "SKILL.md"
        super().__init__(
            name="complex_query",
            llm=llm,
            tool_manager=tool_manager,
            skill_md_path=str(_md),
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the complex query skill graph"""
        from langgraph.graph import MessagesState
        from typing_extensions import TypedDict
        from typing import Annotated
        from operator import add
        
        # Define custom state extending MessagesState
        class ComplexQueryGraphState(TypedDict):
            messages: Annotated[list, add_messages]
            tables: list
            table_schema: str
            query_plan: list  # List of step dicts
            step_results: dict  # Dict of step_id -> result
            plan_completed: bool
            # Retrieval stats (injected by dual_tower_retrieve node)
            retrieval_stats: dict
            # Session plan tracking
            task_id: str
            constraints: list   # user-defined hard constraints
            thread_id: str      # conversation thread id (injected by _make_skill_node)
        
        # Use custom state
        graph = StateGraph(ComplexQueryGraphState)
        
        # Add nodes
        graph.add_node("list_tables", self.common.create_list_tables_node())
        graph.add_node("get_schema", self.common.create_get_schema_node())
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute_step", self._execute_step_node)
        graph.add_node("aggregate", self._aggregate_node)
        graph.add_node("judge", self._judge_node)
        
        # Build flow — insert retrieval node if retriever is configured
        graph.add_edge(START, "list_tables")
        graph.add_edge("list_tables", "get_schema")

        if self._retriever is not None:
            graph.add_node("dual_tower_retrieve", self._dual_tower_retrieve_node)
            graph.add_edge("get_schema", "dual_tower_retrieve")
            graph.add_edge("dual_tower_retrieve", "plan")
        else:
            graph.add_edge("get_schema", "plan")
        
        # Conditional: plan decides whether to execute steps or end
        graph.add_conditional_edges(
            "plan",
            self._should_execute_steps,
            {
                "execute": "execute_step",
                "end": END
            }
        )
        
        graph.add_edge("execute_step", "aggregate")
        
        # Conditional: judge decides whether to continue or end
        graph.add_conditional_edges(
            "judge",
            self._should_continue,
            {
                "continue": "plan",
                "end": END
            }
        )
        
        graph.add_edge("aggregate", "judge")
        
        return graph.compile()
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        """Extract the most recent HumanMessage content from state."""
        from langchain_core.messages import HumanMessage as HM
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HM):
                return msg.content
        return messages[0].content if messages else ""

    def _dual_tower_retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """双塔检索节点：用 Milvus + Steiner Tree 剪枝 schema。"""
        user_question = self._get_user_question(state)
        full_schema = state.get("table_schema", "")

        logger.info(f"[DualTower] Retrieving schema for: {user_question[:80]}")

        try:
            result = self._retriever.retrieve(user_question)

            # Replace full schema with pruned schema
            stats = {
                "full_chars": result.full_schema_chars,
                "pruned_chars": result.pruned_schema_chars,
                "saved_chars": result.char_saved,
                "saved_tokens_est": result.estimated_token_saved,
                "reduction_pct": round(result.reduction_pct, 1),
                "join_path_tables": result.join_path_tables,
                "retrieval_ms": round(result.retrieval_ms, 1),
            }
            logger.info(result.summary())

            return {
                "table_schema": result.pruned_schema,
                "retrieval_stats": stats,
            }

        except Exception as e:
            logger.warning(f"[DualTower] Retrieval node failed: {e}, using full schema")
            return {
                "table_schema": full_schema,
                "retrieval_stats": {"error": str(e)},
            }

    def _plan_node(self, state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
        """
        Analyze the question and generate a multi-step query plan.
        Creates / updates the session plan file for task tracking.
        """
        logger.info("[ComplexQuery] Planning multi-step query")
        
        table_schema = state.get("table_schema", "")
        query_plan = state.get("query_plan", [])
        task_id = state.get("task_id", "")
        thread_id = state.get("thread_id", "") or (config or {}).get("configurable", {}).get("thread_id", "")
        
        user_question = self._get_user_question(state)

        if query_plan:
            logger.info("[ComplexQuery] Plan already exists, skipping planning")
            return {}
        
        # ── Session plan: inject prior progress if resuming ────────────────
        plan_context = ""
        if self._plan_manager and task_id:
            plan_context = self._plan_manager.format_for_llm(task_id)
            if plan_context:
                logger.info(f"[SessionPlan] Injecting plan progress into LLM context")

        plan_context_inner = plan_context if plan_context else "（无）"
        constraints_block = self._build_constraints_block(state.get("constraints", []))

        system_prompt = f"""[Role & Policies]
您是一个 MySQL 数据库的多步查询规划专家。
严格遵守 MySQL 兼容规则，输出纯 JSON，不加任何解释或 Markdown 包装。

[Task]
分析复杂问题，拆分为多个独立的 SQL 子查询步骤，生成可执行的查询计划。

[Environment]
- 数据库方言：MySQL
- 子查询步骤间依赖通过占位符 {{step_N_results}} 传递（只能配合 IN 运算符）
- MySQL 限制：IN 子查询内不允许使用 LIMIT
- 保持查询简洁，避免深层嵌套子查询

[Evidence]
可用表和 Schema：
{table_schema}

[Context]
{plan_context_inner}

[Output]
输出格式（纯 JSON）：
{{
    "steps": [
        {{
            "step_id": 1,
            "description": "获取 TOP 3 店铺类型 ID",
            "query": "SELECT id, name FROM tb_shop_type ORDER BY score DESC LIMIT 3",
            "depends_on": []
        }},
        {{
            "step_id": 2,
            "description": "统计各类型店铺数量",
            "query": "SELECT type_id, COUNT(*) FROM tb_shop WHERE type_id IN {{step_1_results}} GROUP BY type_id",
            "depends_on": [1]
        }}
    ]
}}

占位符规则：
✅ WHERE column IN {{step_N_results}}
❌ FROM ({{step_N_results}}) — 不允许
❌ WHERE id = {{step_N_results}} — 不允许

如果问题较简单（单表单查询），输出：{{"simple": true, "reason": "..."}}{constraints_block}
"""
        
        plan_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {user_question}\n\nGenerate a query plan.")
        ]
        
        response = self.llm.invoke(plan_messages)
        
        import json
        try:
            plan_data = json.loads(response.content)
            
            if plan_data.get("simple"):
                logger.info(f"[ComplexQuery] Question is simple: {plan_data.get('reason')}")
                new_message = AIMessage(content=f"This question is too simple for complex query handling. Reason: {plan_data.get('reason')}")
                return {
                    "messages": [new_message],
                    "query_plan": [],
                    "plan_completed": True
                }
            
            steps = plan_data.get("steps", [])
            logger.info(f"[ComplexQuery] Generated plan with {len(steps)} steps")

            # ── Session plan: create plan file ─────────────────────────────
            new_task_id = task_id
            if self._plan_manager:
                if not task_id:
                    new_task_id = self._plan_manager.new_task_id()
                self._plan_manager.create_plan(
                    task_id=new_task_id,
                    title=user_question[:80],
                    description=user_question,
                    skill="complex_query",
                    steps=steps,
                    thread_id=thread_id,
                )
                logger.info(f"[SessionPlan] Plan file: {self._plan_manager.get_plan_path(new_task_id)}")
            
            new_message = AIMessage(content=f"Query plan generated with {len(steps)} steps:\n" + 
                                           "\n".join([f"{s['step_id']}. {s['description']}" for s in steps]))
            
            return {
                "messages": [new_message],
                "query_plan": steps,
                "step_results": {},
                "plan_completed": False,
                "task_id": new_task_id,
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"[ComplexQuery] Failed to parse plan: {e}")
            error_message = AIMessage(content=f"Failed to generate query plan: {str(e)}")
            return {
                "messages": [error_message],
                "query_plan": [],
                "plan_completed": True
            }
    
    def _should_execute_steps(self, state: Dict[str, Any]) -> Literal["execute", "end"]:
        """Decide whether to execute steps or end"""
        query_plan = state.get("query_plan", [])
        plan_completed = state.get("plan_completed", False)
        
        if not query_plan or plan_completed:
            return "end"
        return "execute"
    
    def _execute_step_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query steps with dependency resolution
        
        This node executes steps that have all dependencies satisfied,
        replacing placeholders with actual results from previous steps.
        """
        logger.info("[ComplexQuery] Executing query steps with dependency resolution")
        
        query_plan = state.get("query_plan", [])
        step_results = state.get("step_results", {})
        
        # Find steps that are ready to execute (dependencies completed)
        ready_steps = []
        for step in query_plan:
            step_id = step["step_id"]
            
            # Skip if already executed
            if step_id in step_results:
                continue
            
            # Check dependencies
            depends_on = step.get("depends_on", [])
            deps_ready = all(dep_id in step_results for dep_id in depends_on)
            
            if deps_ready:
                ready_steps.append(step)
        
        logger.info(f"[ComplexQuery] Found {len(ready_steps)} ready steps to execute")
        
        # Execute ready steps
        for step in ready_steps:
            step_id = step["step_id"]
            query = step["query"]

            # ── 用户禁令守卫：执行前硬性拦截 ──────────────────────────────────
            from agent.constraint_guard import check_constraints, build_block_message
            from agent.sql_errors import UserPolicyError
            try:
                check_constraints(query, state.get("constraints", []))
            except UserPolicyError as ce:
                logger.warning(f"[ComplexQuery] Blocked by constraint: {ce.matched_keyword!r}")
                from langchain_core.messages import AIMessage
                messages = list(state.get("messages", []))
                messages.append(AIMessage(content=build_block_message(ce)))
                return {"messages": messages, "plan_completed": True, "step_results": state.get("step_results", {})}
            task_id = state.get("task_id", "")
            
            # ── Session plan: mark step in_progress ───────────────────────
            if self._plan_manager and task_id:
                self._plan_manager.update_step(task_id, step_id, "in_progress")
            
            # Replace placeholders with actual results from dependencies
            query = self._resolve_query_placeholders(query, step.get("depends_on", []), step_results)
            perf_opt_result = self._perf_analyzer.optimize_sql(
                query,
                constraints=state.get("constraints", []),
            )
            query = perf_opt_result.final_sql
            perf_analysis = perf_opt_result.final_analysis
            logger.info("[ComplexQuery][SQLPerf][step=%s][before] %s", step_id, perf_opt_result.original_analysis.summary)
            logger.info("[ComplexQuery][SQLPerf][step=%s][after] %s", step_id, perf_analysis.summary)
            if perf_opt_result.optimized:
                logger.info("[ComplexQuery][SQLPerf][step=%s] SQL optimized and replaced", step_id)
            
            # ── SQL 执行前用户确认 ────────────────────────────────────────────
            if self.confirm_enabled and query:
                from agent.sql_confirm import prompt_sql_confirmation, build_skip_message
                from agent.types import SQLSkippedByUser
                action, reason = prompt_sql_confirmation(query)
                if action == "skip":
                    skip_content = build_skip_message(query, reason)
                    logger.info(f"[ComplexQuery] Step {step_id} skipped by user")
                    step_results[step_id] = {
                        "step_id": step_id,
                        "description": step["description"],
                        "query": query,
                        "original_query": step["query"],
                        "result": skip_content,
                        "success": False,
                        "skipped": True,
                        "retries": 0,
                        "performance": perf_analysis.to_dict(),
                        "performance_optimization": perf_opt_result.to_dict(),
                    }
                    if self._plan_manager and task_id:
                        self._plan_manager.update_step(task_id, step_id, "skipped")
                    continue

            # ── 统一执行管线：直执失败后自动走纠错 ───────────────────────────
            started_at = time.perf_counter()
            pipeline_result = run_sql_execution_pipeline(
                sql=query,
                query_tool=self.tool_manager.get_query_tool(),
                db_manager=self.db_manager,
                llm=self.llm,
                perf_analyzer=self._perf_analyzer,
                constraints=state.get("constraints", []),
                context_label=f"[ComplexQuery] step {step_id}",
                correction_max_retries=self._correction_max_retries,
                precomputed_optimization=perf_opt_result,
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info("[ComplexQuery][Pipeline][step=%s] decision=%s", step_id, pipeline_result.decision)

            if pipeline_result.success:
                result = pipeline_result.result
                step_results[step_id] = {
                    "step_id": step_id,
                    "description": step["description"],
                    "query": pipeline_result.final_sql,
                    "original_query": step["query"],
                    "result": result,
                    "success": True,
                    "elapsed_ms": elapsed_ms,
                    "retries": pipeline_result.retries,
                    "correction_trace": pipeline_result.correction_trace,
                    "pipeline_decision": pipeline_result.decision,
                    "performance": perf_analysis.to_dict(),
                    "performance_optimization": perf_opt_result.to_dict(),
                }
                logger.info(f"[ComplexQuery] Step {step_id} executed successfully (retries={pipeline_result.retries})")

                # ── Emit SQL step event for live frontend ─────────────────
                from agent import sql_step_emitter
                sql_step_emitter.emit(
                    str(step_id),
                    step["description"],
                    pipeline_result.final_sql,
                    performance=perf_analysis.to_dict(),
                    optimization=perf_opt_result.to_dict(),
                    elapsed_ms=elapsed_ms,
                )

                # ── Session plan: mark step done ──────────────────────────
                if self._plan_manager and task_id:
                    result_preview = str(result)[:200] if result else ""
                    self._plan_manager.update_step(
                        task_id, step_id, "done",
                        sql=pipeline_result.final_sql,
                        elapsed_ms=elapsed_ms,
                        result_summary=result_preview,
                        notes=(
                            (
                                f"耗时: {elapsed_ms}ms"
                                + f" | 评分: {perf_opt_result.original_analysis.score}->{perf_analysis.score}"
                                + (f" | {perf_analysis.summary}" if perf_analysis else "")
                                + (" | 已自动优化" if perf_opt_result.optimized else "")
                            )
                        )[:200],
                    )
            else:
                error_str = pipeline_result.error
                logger.error(f"[ComplexQuery] Step {step_id} failed after correction: {error_str}")
                step_results[step_id] = {
                    "step_id": step_id,
                    "description": step["description"],
                    "query": pipeline_result.final_sql,
                    "original_query": step["query"],
                    "result": f"Error: {error_str}",
                    "success": False,
                    "retries": pipeline_result.retries,
                    "correction_trace": pipeline_result.correction_trace,
                    "pipeline_decision": pipeline_result.decision,
                    "performance": perf_analysis.to_dict(),
                    "performance_optimization": perf_opt_result.to_dict(),
                }
                # ── Session plan: mark step failed ────────────────────────
                if self._plan_manager and task_id:
                    self._plan_manager.update_step(
                        task_id, step_id, "failed",
                        sql=pipeline_result.final_sql,
                        error=error_str,
                    )
                    self._plan_manager.add_note(
                        task_id, "blocker",
                        f"Step {step_id} 执行失败（纠错后仍失败）",
                        f"SQL: {pipeline_result.final_sql[:150]}\n错误: {error_str[:150]}",
                    )
        
        # Add SQL step messages for history reconstruction
        sql_msgs = []
        for sr in step_results.values():
            if sr.get("success") and sr.get("query"):
                sql_msgs.append(AIMessage(
                    content=f"__sql__:{sr['step_id']}:{sr['description']}:{sr['query']}"
                ))
                sql_msgs.append(AIMessage(
                    content="__sqlmeta__:" + json.dumps({
                        "step_id": str(sr["step_id"]),
                        "label": sr.get("description", "SQL 查询"),
                        "sql": sr["query"],
                        "elapsed_ms": sr.get("elapsed_ms"),
                        "performance": sr.get("performance"),
                        "optimization": sr.get("performance_optimization"),
                    }, ensure_ascii=False)
                ))
        
        return {"step_results": step_results, "messages": sql_msgs}
    
    def _resolve_query_placeholders(self, query: str, depends_on: List[int], step_results: Dict) -> str:
        """
        Replace placeholders like {step_N_results} with actual values
        
        Args:
            query: SQL query with placeholders
            depends_on: List of step IDs this query depends on
            step_results: Dictionary of completed step results
            
        Returns:
            Query with placeholders replaced by actual values
        """
        import re
        import ast
        
        for dep_id in depends_on:
            placeholder = f"{{step_{dep_id}_results}}"
            
            if placeholder in query:
                # Get the result from the dependency step
                dep_step = step_results.get(dep_id, {})
                dep_result = dep_step.get("result", [])
                
                logger.info(f"[ComplexQuery] Resolving placeholder for step {dep_id}")
                logger.info(f"[ComplexQuery] Raw result type: {type(dep_result)}, value: {dep_result}")
                
                # If result is a string, try to parse it as a Python literal
                if isinstance(dep_result, str):
                    try:
                        dep_result = ast.literal_eval(dep_result)
                        logger.info(f"[ComplexQuery] Parsed string to: {type(dep_result)}, value: {dep_result}")
                    except (ValueError, SyntaxError) as e:
                        logger.error(f"[ComplexQuery] Failed to parse result string: {e}")
                        query = query.replace(placeholder, "(NULL)")
                        continue
                
                if dep_result is not None and dep_result != []:
                    # Extract IDs from tuples (assuming first element is ID)
                    if isinstance(dep_result, list):
                        if len(dep_result) > 0:
                            if isinstance(dep_result[0], tuple):
                                # Extract first element from each tuple
                                ids = [str(row[0]) for row in dep_result]
                                logger.info(f"[ComplexQuery] Extracted IDs from tuples: {ids}")
                            else:
                                # Direct list of values
                                ids = [str(val) for val in dep_result]
                                logger.info(f"[ComplexQuery] Direct list of IDs: {ids}")
                            
                            # Replace placeholder with comma-separated IDs in parentheses
                            id_list = "(" + ", ".join(ids) + ")"
                            query = query.replace(placeholder, id_list)
                            logger.info(f"[ComplexQuery] Replaced {placeholder} with: {id_list}")
                        else:
                            logger.warning(f"[ComplexQuery] Empty result list for step {dep_id}")
                            query = query.replace(placeholder, "(NULL)")
                    else:
                        logger.warning(f"[ComplexQuery] Result is not a list for step {dep_id}, type: {type(dep_result)}")
                        query = query.replace(placeholder, "(NULL)")
                else:
                    logger.warning(f"[ComplexQuery] No result found for step {dep_id}")
                    query = query.replace(placeholder, "(NULL)")
        
        logger.info(f"[ComplexQuery] Final query: {query}")
        return query
    
    def _aggregate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and format all step results"""
        logger.info("[ComplexQuery] Aggregating step results")
        
        step_results = state.get("step_results", {})
        query_plan = state.get("query_plan", [])
        
        # Format results
        result_text = "Query execution results:\n\n"
        
        for step in query_plan:
            step_id = step["step_id"]
            result = step_results.get(step_id, {})
            
            result_text += f"Step {step_id}: {step['description']}\n"
            
            if result.get("skipped"):
                result_text += f"Status: 用户跳过此步骤（无查询结果）\n\n"
            elif result.get("success"):
                result_text += f"Result: {result['result']}\n\n"
            else:
                result_text += f"Error: {result.get('error', 'Unknown error')}\n\n"
        
        new_message = AIMessage(content=result_text)
        
        return {"messages": [new_message]}
    
    def _judge_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all steps are completed and mark session plan accordingly"""
        logger.info("[ComplexQuery] Judging plan completion")
        
        query_plan = state.get("query_plan", [])
        step_results = state.get("step_results", {})
        task_id = state.get("task_id", "")
        
        total_steps = len(query_plan)
        completed_steps = len(step_results)
        all_completed = total_steps == completed_steps
        
        logger.info(f"[ComplexQuery] Completion: {completed_steps}/{total_steps} steps")
        
        # ── Session plan: mark overall task complete ───────────────────────
        if self._plan_manager and task_id and all_completed:
            success = all(r.get("success", False) for r in step_results.values())
            self._plan_manager.mark_complete(task_id, success=success)
            logger.info(f"[SessionPlan] Task {task_id} marked {'done' if success else 'failed'}")
        
        return {"plan_completed": all_completed}
    
    def _should_continue(self, state: Dict[str, Any]) -> Literal["continue", "end"]:
        """Decide whether to continue or end"""
        plan_completed = state.get("plan_completed", False)
        
        if plan_completed:
            return "end"
        return "continue"
