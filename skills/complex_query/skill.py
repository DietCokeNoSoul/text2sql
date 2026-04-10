"""
Complex Query Skill Implementation

Plan-Execute pattern:
1. list_tables → get_schema (reuse common nodes)
2. plan → analyze question, generate multi-step query plan
3. execute_step → parallel execution using Send API
4. aggregate → collect all step results
5. judge → check if all steps completed, loop if needed
"""

import logging
from typing import Any, Dict, List, Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.types import Send

from agent.tools import SQLToolManager
from agent.database import SQLDatabaseManager
from agent.skills.base import BaseSkill
from agent.skills.states import ComplexQueryState

logger = logging.getLogger(__name__)


class ComplexQuerySkill(BaseSkill):
    """
    Complex Query Skill - Plan-Execute pattern with parallel execution
    
    Flow:
        list_tables → get_schema → plan → execute_steps (parallel) → aggregate → judge
        
    Uses Send API for parallel execution of sub-queries.
    """
    
    def __init__(self, llm: BaseChatModel, tool_manager: SQLToolManager, db_manager: SQLDatabaseManager):
        self.db_manager = db_manager
        self.simple_query_tool = None  # Will be set by main graph if needed
        
        super().__init__(
            name="complex_query",
            llm=llm,
            tool_manager=tool_manager,
            description="处理需要多步骤分解的复杂查询（Plan-Execute模式）"
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
        
        # Use custom state
        graph = StateGraph(ComplexQueryGraphState)
        
        # Add nodes
        graph.add_node("list_tables", self.common.create_list_tables_node())
        graph.add_node("get_schema", self.common.create_get_schema_node())
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute_step", self._execute_step_node)
        graph.add_node("aggregate", self._aggregate_node)
        graph.add_node("judge", self._judge_node)
        
        # Build flow
        graph.add_edge(START, "list_tables")
        graph.add_edge("list_tables", "get_schema")
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
    
    def _plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the question and generate a multi-step query plan
        
        Each step should be a sub-query that can be executed independently.
        """
        logger.info("[ComplexQuery] Planning multi-step query")
        
        messages = state.get("messages", [])
        table_schema = state.get("table_schema", "")
        query_plan = state.get("query_plan", [])
        
        # Get the original question
        user_question = messages[0].content if messages else ""
        
        # Check if we already have a plan
        if query_plan:
            logger.info("[ComplexQuery] Plan already exists, skipping planning")
            return {}
        
        # Create planning prompt
        system_prompt = f"""You are a SQL query planner for MySQL database. Given a complex question, break it down into multiple independent sub-queries.

Available tables and schema:
{table_schema}

**CRITICAL: How to use placeholders for dependent steps**:
- Use `{{step_N_results}}` to reference results from step N
- This placeholder will be replaced with VALUES like `(1, 2, 3)` - a list of IDs
- Use it ONLY with IN operator: `WHERE column IN {{step_N_results}}`
- DO NOT treat it as a table or subquery

**WRONG usage** (will cause syntax errors):
❌ `SELECT id FROM ({{step_1_results}}) AS t`
❌ `JOIN ({{step_1_results}}) t ON ...`
❌ `WHERE id = {{step_1_results}}`

**CORRECT usage**:
✅ `WHERE type_id IN {{step_1_results}}`
✅ `WHERE shop_id IN {{step_1_results}}`
✅ `WHERE id IN {{step_1_results}} AND status = 'active'`

**MySQL Compatibility Rules**:
1. DO NOT use "LIMIT" inside subqueries with IN/ALL/ANY/SOME
2. Keep queries simple and direct
3. Avoid deeply nested subqueries

**Output format (JSON)**:
{{
    "steps": [
        {{
            "step_id": 1, 
            "description": "Get top 3 shop type IDs", 
            "query": "SELECT id, name FROM tb_shop_type ORDER BY score DESC LIMIT 3",
            "depends_on": []
        }},
        {{
            "step_id": 2, 
            "description": "Count shops for each type", 
            "query": "SELECT type_id, COUNT(*) FROM tb_shop WHERE type_id IN {{step_1_results}} GROUP BY type_id",
            "depends_on": [1]
        }}
    ]
}}

If the question is simple (single table, single query), output:
{{"simple": true, "reason": "..."}}
"""
        
        plan_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {user_question}\n\nGenerate a query plan.")
        ]
        
        response = self.llm.invoke(plan_messages)
        
        # Parse the plan
        import json
        try:
            plan_data = json.loads(response.content)
            
            if plan_data.get("simple"):
                # Question is too simple for complex query skill
                logger.info(f"[ComplexQuery] Question is simple: {plan_data.get('reason')}")
                new_message = AIMessage(content=f"This question is too simple for complex query handling. Reason: {plan_data.get('reason')}")
                return {
                    "messages": [new_message],
                    "query_plan": [],
                    "plan_completed": True
                }
            
            steps = plan_data.get("steps", [])
            logger.info(f"[ComplexQuery] Generated plan with {len(steps)} steps")
            
            new_message = AIMessage(content=f"Query plan generated with {len(steps)} steps:\n" + 
                                           "\n".join([f"{s['step_id']}. {s['description']}" for s in steps]))
            
            return {
                "messages": [new_message],
                "query_plan": steps,
                "step_results": {},
                "plan_completed": False
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
            
            # Replace placeholders with actual results from dependencies
            query = self._resolve_query_placeholders(query, step.get("depends_on", []), step_results)
            
            try:
                # Execute query using tool
                query_tool = self.tool_manager.get_query_tool()
                result = query_tool.invoke({"query": query})
                
                step_results[step_id] = {
                    "step_id": step_id,
                    "description": step["description"],
                    "query": query,
                    "original_query": step["query"],
                    "result": result,
                    "success": True
                }
                logger.info(f"[ComplexQuery] Step {step_id} executed successfully")
                
            except Exception as e:
                logger.error(f"[ComplexQuery] Step {step_id} failed: {e}")
                step_results[step_id] = {
                    "step_id": step_id,
                    "description": step["description"],
                    "query": query,
                    "original_query": step["query"],
                    "error": str(e),
                    "success": False
                }
        
        return {"step_results": step_results}
    
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
            
            if result.get("success"):
                result_text += f"Result: {result['result']}\n\n"
            else:
                result_text += f"Error: {result.get('error', 'Unknown error')}\n\n"
        
        new_message = AIMessage(content=result_text)
        
        return {"messages": [new_message]}
    
    def _judge_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all steps are completed"""
        logger.info("[ComplexQuery] Judging plan completion")
        
        query_plan = state.get("query_plan", [])
        step_results = state.get("step_results", {})
        
        # Check if all steps are completed
        total_steps = len(query_plan)
        completed_steps = len(step_results)
        
        all_completed = total_steps == completed_steps
        
        logger.info(f"[ComplexQuery] Completion: {completed_steps}/{total_steps} steps")
        
        return {"plan_completed": all_completed}
    
    def _should_continue(self, state: Dict[str, Any]) -> Literal["continue", "end"]:
        """Decide whether to continue or end"""
        plan_completed = state.get("plan_completed", False)
        
        if plan_completed:
            return "end"
        return "continue"
