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
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langgraph.graph.message import add_messages

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
        system_prompt = f"""You are a SQL query planner. Given a complex question, break it down into multiple independent sub-queries.

Available tables and schema:
{table_schema}

Rules:
1. Each step should be a simple, self-contained query
2. Steps can be executed in parallel if they don't depend on each other
3. Use clear, descriptive step names
4. Keep queries focused and specific

Output format (JSON):
{{
    "steps": [
        {{"step_id": 1, "description": "...", "query": "...", "depends_on": []}},
        {{"step_id": 2, "description": "...", "query": "...", "depends_on": [1]}}
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
        Execute query steps in parallel using Send API
        
        This node will be called once per step via Send.
        """
        logger.info("[ComplexQuery] Executing query steps in parallel")
        
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
            
            try:
                # Execute query using common node
                query_tool = self.tool_manager.get_query_tool()
                result = query_tool.invoke({"query": query})
                
                step_results[step_id] = {
                    "step_id": step_id,
                    "description": step["description"],
                    "query": query,
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
                    "error": str(e),
                    "success": False
                }
        
        return {"step_results": step_results}
    
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
