"""
Skill-based Main Graph Builder

Integrates three Skills (Simple Query, Complex Query, Data Analysis)
with intelligent routing based on query complexity.
"""

import logging
from typing import Any, Dict, Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.config import AgentConfig
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager

# Note: Skills are imported lazily in __init__ to avoid circular imports

logger = logging.getLogger(__name__)


class SkillBasedGraphBuilder:
    """
    Build main graph with Skill-based architecture.
    
    Flow:
        user_input → query_router → [simple_skill | complex_skill | analysis_skill] → output
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm: BaseChatModel,
        db_manager: SQLDatabaseManager,
        tool_manager: SQLToolManager,
        checkpointer: BaseCheckpointSaver = None
    ):
        self.config = config
        self.llm = llm
        self.db_manager = db_manager
        self.tool_manager = tool_manager
        self.checkpointer = checkpointer
        
        # Lazy import Skills to avoid circular imports
        from skills.simple_query import SimpleQuerySkill
        from skills.complex_query import ComplexQuerySkill
        from skills.data_analysis import DataAnalysisSkill
        
        # Initialize Skills
        logger.info("Initializing Skills...")
        self.simple_skill = SimpleQuerySkill(llm, tool_manager, db_manager)
        self.complex_skill = ComplexQuerySkill(llm, tool_manager, db_manager)
        self.analysis_skill = DataAnalysisSkill(llm, tool_manager, db_manager, config=config)
        
        logger.info("Skills initialized successfully")
    
    def build(self) -> StateGraph:
        """Build the main skill-based graph."""
        from typing_extensions import TypedDict
        from typing import Annotated
        
        # Define main graph state
        class MainGraphState(TypedDict):
            messages: Annotated[list, add_messages]
            query_type: str  # "simple", "complex", "analysis"
            skill_result: dict  # Result from selected skill
        
        logger.info("Building Skill-based main graph")
        
        graph = StateGraph(MainGraphState)
        
        # Add router node
        graph.add_node("query_router", self._query_router_node)
        
        # Add skill nodes
        graph.add_node("simple_skill", self._simple_skill_node)
        graph.add_node("complex_skill", self._complex_skill_node)
        graph.add_node("analysis_skill", self._analysis_skill_node)
        
        # Define routing flow
        graph.add_edge(START, "query_router")
        
        # Conditional routing based on query type
        graph.add_conditional_edges(
            "query_router",
            self._route_to_skill,
            {
                "simple": "simple_skill",
                "complex": "complex_skill",
                "analysis": "analysis_skill"
            }
        )
        
        # All skills end at END
        graph.add_edge("simple_skill", END)
        graph.add_edge("complex_skill", END)
        graph.add_edge("analysis_skill", END)
        
        # Compile with checkpointer if provided
        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()
    
    def _query_router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route query to appropriate Skill based on complexity analysis.
        
        Classification:
        - Simple: Single table, basic conditions, no aggregation
        - Complex: Multiple tables, JOINs, subqueries, multi-step logic
        - Analysis: Requires insights, trends, visualizations, reports
        """
        logger.info("[Router] Analyzing query complexity")
        
        messages = state.get("messages", [])
        user_question = messages[0].content if messages else ""
        
        # Classification prompt
        system_prompt = """You are a query complexity classifier. Analyze the user's question and classify it into ONE category:

**SIMPLE** - Single table query with basic filters
Examples:
- "List all users"
- "Show me the first 10 products"
- "Find users where status = 'active'"
- "Get shop with ID = 5"

**COMPLEX** - Multi-step query requiring planning
Examples:
- "Find top 3 shops by voucher count with average values"
- "Compare sales across regions and product categories"
- "Show users with most blog posts AND comments"
- "Calculate conversion rate by shop and time period"

**ANALYSIS** - Requires insights, trends, or recommendations
Examples:
- "Analyze user engagement trends"
- "What factors drive high sales?"
- "Identify underperforming categories and suggest improvements"
- "Create a report on platform growth"
- "Visualize the distribution of..."

Output ONLY the category name: "simple", "complex", or "analysis"
"""
        
        classification_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {user_question}\n\nClassify this query.")
        ]
        
        response = self.llm.invoke(classification_messages)
        classification = response.content.strip().lower()
        
        # Validate classification
        if classification not in ["simple", "complex", "analysis"]:
            logger.warning(f"[Router] Invalid classification: {classification}, defaulting to 'simple'")
            classification = "simple"
        
        logger.info(f"[Router] Query classified as: {classification}")
        
        new_message = AIMessage(
            content=f"Query Type: {classification.upper()}"
        )
        
        return {
            "messages": [new_message],
            "query_type": classification
        }
    
    def _route_to_skill(self, state: Dict[str, Any]) -> Literal["simple", "complex", "analysis"]:
        """Determine which skill to route to based on query_type."""
        query_type = state.get("query_type", "simple")
        logger.info(f"[Router] Routing to {query_type} skill")
        return query_type
    
    def _simple_skill_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Simple Query Skill."""
        logger.info("[Main] Executing Simple Query Skill")
        
        # Invoke skill
        result = self.simple_skill.invoke(state)
        
        return {
            "messages": result.get("messages", []),
            "skill_result": result
        }
    
    def _complex_skill_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Complex Query Skill."""
        logger.info("[Main] Executing Complex Query Skill")
        
        # Invoke skill
        result = self.complex_skill.invoke(state)
        
        return {
            "messages": result.get("messages", []),
            "skill_result": result
        }
    
    def _analysis_skill_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Data Analysis Skill."""
        logger.info("[Main] Executing Data Analysis Skill")
        
        # Invoke skill
        result = self.analysis_skill.invoke(state)
        
        return {
            "messages": result.get("messages", []),
            "skill_result": result
        }


def create_skill_based_graph(
    config: AgentConfig,
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver = None
) -> StateGraph:
    """
    Create the Skill-based SQL Agent graph.
    
    Args:
        config: Agent configuration
        llm: Language model instance
        checkpointer: Optional checkpointer for memory
        
    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating Skill-based SQL Agent graph")
    
    # Initialize managers
    db_manager = SQLDatabaseManager(config.database)
    tool_manager = SQLToolManager(db_manager, llm)
    
    # Build graph
    builder = SkillBasedGraphBuilder(
        config=config,
        llm=llm,
        db_manager=db_manager,
        tool_manager=tool_manager,
        checkpointer=checkpointer
    )
    
    graph = builder.build()
    
    logger.info("Skill-based SQL Agent graph created successfully")
    return graph
