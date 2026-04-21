"""
Data Analysis Skill Implementation

7-step comprehensive data analysis workflow:
1. understand_goal → Understand user's analysis objective
2. explore_data → Explore database structure and statistics
3. plan_analysis → Generate detailed analysis plan
4. generate_sql_queries → Create SQL queries for analysis
5. analyze_results → Analyze query results and extract insights
6. visualize → Generate visualization recommendations
7. generate_report → Create comprehensive analysis report
"""

import logging
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START, add_messages

from agent.config import AgentConfig, OutputConfig
from agent.tools import SQLToolManager
from agent.database import SQLDatabaseManager
from agent.skills.base import BaseSkill

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
    ):
        self.db_manager = db_manager
        self._output_config: OutputConfig = config.output if config else OutputConfig()
        self._plan_manager = plan_manager
        
        super().__init__(
            name="data_analysis",
            llm=llm,
            tool_manager=tool_manager,
            description="执行完整的数据分析流程，包含洞察发现和可视化建议"
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
            report: str
            task_id: str        # session plan tracking
        
        graph = StateGraph(DataAnalysisState)
        
        # Add 7 specialized nodes
        graph.add_node("understand_goal", self._understand_goal)
        graph.add_node("explore_data", self._explore_data)
        graph.add_node("plan_analysis", self._plan_analysis)
        graph.add_node("generate_queries", self._generate_queries)
        graph.add_node("analyze_results", self._analyze_results)
        graph.add_node("visualize", self._visualize)
        graph.add_node("generate_report", self._generate_report)
        
        # Build linear flow
        graph.add_edge(START, "understand_goal")
        graph.add_edge("understand_goal", "explore_data")
        graph.add_edge("explore_data", "plan_analysis")
        graph.add_edge("plan_analysis", "generate_queries")
        graph.add_edge("generate_queries", "analyze_results")
        graph.add_edge("analyze_results", "visualize")
        graph.add_edge("visualize", "generate_report")
        graph.add_edge("generate_report", END)
        
        return graph.compile()
    
    # ============ Node 1: Understand Goal ============
    
    def _understand_goal(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
        user_question = messages[0].content if messages else ""
        
        system_prompt = """You are a data analyst. Analyze the user's request and extract:

1. Primary objective - What is the main question?
2. Key metrics - What should be measured?
3. Dimensions - What should be grouped/compared?
4. Filters - Any time ranges, conditions, or limits?
5. Output expectations - What format should results be in?

**CRITICAL**: Output ONLY valid JSON, no markdown, no explanation.

Output format:
{
    "objective": "...",
    "metrics": ["...", "..."],
    "dimensions": ["...", "..."],
    "filters": {},
    "output_format": "..."
}

Return ONLY the JSON object, nothing else.
"""
        
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
                    steps=[
                        {"step_id": 1, "description": "understand_goal: 理解分析目标", "depends_on": []},
                        {"step_id": 2, "description": "explore_data: 探索数据库结构", "depends_on": [1]},
                        {"step_id": 3, "description": "plan_analysis: 制定分析计划", "depends_on": [2]},
                        {"step_id": 4, "description": "generate_queries: 生成SQL查询", "depends_on": [3]},
                        {"step_id": 5, "description": "analyze_results: 分析结果洞察", "depends_on": [4]},
                        {"step_id": 6, "description": "visualize: 生成可视化建议", "depends_on": [5]},
                        {"step_id": 7, "description": "generate_report: 生成分析报告", "depends_on": [6]},
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
        
        system_prompt = f"""You are a data analyst creating an analysis plan.

Analysis Goal:
{analysis_goal}

Available Data:
{data_summary}

Create a detailed analysis plan with:
1. Analysis steps (sequential)
2. Required calculations/aggregations
3. Expected insights
4. Potential visualizations

**CRITICAL**: Output ONLY valid JSON, no markdown, no explanation.

Output format:
{{
    "steps": [
        {{"id": 1, "description": "...", "tables": ["..."], "calculations": ["..."]}},
        {{"id": 2, "description": "...", "tables": ["..."], "calculations": ["..."]}}
    ],
    "expected_insights": ["...", "..."],
    "visualization_suggestions": ["...", "..."]
}}

Return ONLY the JSON object, nothing else.
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
            
            system_prompt = f"""Generate a MySQL-compatible SQL query for this analysis step.

Step: {description}
Required Calculations: {', '.join(calculations) if calculations else 'as needed'}

Available Tables and Schemas:
{all_schemas}

**CRITICAL**: Output ONLY the SQL query. No markdown, no explanation, no code blocks.
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
        
        return {
            "messages": [new_message],
            "sql_queries": sql_queries
        }
    
    def _generate_queries_from_goal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: generate queries directly from goal when plan parsing fails."""
        analysis_goal = state.get("analysis_goal", "{}")
        combined_schema = state.get("combined_schema", "")
        
        all_schemas = combined_schema
        
        system_prompt = f"""You are a data analyst. Generate SQL queries to answer the analysis goal.

Analysis Goal:
{analysis_goal}

Available Tables and Schemas:
{all_schemas}

**CRITICAL**: Output ONLY valid JSON, no markdown.

Output format:
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
            
            try:
                # Execute query
                result = query_tool.invoke({"query": sql})
                
                query_results.append({
                    "step_id": step_id,
                    "description": description,
                    "query": sql,
                    "result": result,
                    "success": True
                })
                
                # Generate insight from result
                insight_prompt = f"""Analyze this query result and extract key insights.

Query Purpose: {description}
Result: {result[:500]}

What are the key findings?
"""
                
                insight_messages = [
                    SystemMessage(content="You are a data analyst extracting insights."),
                    HumanMessage(content=insight_prompt)
                ]
                
                insight_response = self.llm.invoke(insight_messages)
                insights.append({
                    "step_id": step_id,
                    "insight": insight_response.content
                })
                
                logger.info(f"[DataAnalysis] Analyzed step {step_id}")
                
            except Exception as e:
                logger.error(f"[DataAnalysis] Query failed for step {step_id}: {e}")
                query_results.append({
                    "step_id": step_id,
                    "description": description,
                    "query": sql,
                    "error": str(e),
                    "success": False
                })
        
        new_message = AIMessage(
            content=f"Executed {len(query_results)} queries, extracted {len(insights)} insights."
        )
        self._step_done(state, 5, f"{len(insights)} insights from {len(query_results)} queries")
        
        return {
            "messages": [new_message],
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
            viz_prompt = f"""Recommend a visualization for this data.

Data Description: {description}
Sample Data: {str(data)[:300]}

Suggest:
1. Best chart type (ONLY one of: bar, pie, line)
2. X-axis description
3. Y-axis description
4. Chart title (short, in Chinese)

Output as JSON ONLY:
{{
    "chart_type": "bar",
    "x_axis": "...",
    "y_axis": "...",
    "title": "...",
    "message": "..."
}}"""

            viz_messages = [
                SystemMessage(content="You are a data visualization expert. Output JSON only."),
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
        
        system_prompt = f"""根据以下分析结果，生成一份完整的中文数据分析报告（Markdown格式）。

分析目标:
{analysis_goal}

查询结果:
{results_text}

关键洞察:
{insights_text}

可视化建议:
{viz_text}

请生成一份结构化的 Markdown 报告，包括：
1. 报告标题和摘要
2. 关键发现（用数据支撑）
3. 可视化建议
4. 结论和建议

直接输出 Markdown 格式的报告，不要包裹在代码块中。
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
        
        new_message = AIMessage(content=report_content)
        self._step_done(state, 7, f"Report generated ({len(report_content)} chars)")
        
        # ── Session plan: mark overall task complete ───────────────────────
        task_id = state.get("task_id", "")
        if self._plan_manager and task_id:
            self._plan_manager.mark_complete(task_id, success=True)
            logger.info(f"[SessionPlan] Task {task_id} complete")
        
        return {
            "messages": [new_message],
            "report": report_content
        }
    
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
