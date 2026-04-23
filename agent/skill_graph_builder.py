"""
Skill-based Main Graph Builder

Integrates three Skills (Simple Query, Complex Query, Data Analysis)
with intelligent routing based on query complexity.
"""

import logging
from typing import Any, Dict, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.config import AgentConfig
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from agent.skills.registry import SkillRegistry

# Note: Skills are imported lazily in __init__ to avoid circular imports

logger = logging.getLogger(__name__)


class SkillBasedGraphBuilder:
    """
    Build main graph with Skill-based architecture.
    
    Flow:
        user_input → query_router → [<skill_name> ...] → output
    
    Skills are discovered from SkillRegistry; descriptions are loaded
    automatically from each Skill's SKILL.md (progressive disclosure).
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

        # Initialize DualTowerRetriever if enabled
        self._retriever = None
        if config.retrieval.enabled:
            try:
                from agent.retrieval import DualTowerRetriever
                self._retriever = DualTowerRetriever(
                    db_manager=db_manager,
                    milvus_host=config.retrieval.milvus_host,
                    milvus_port=config.retrieval.milvus_port,
                    top_k_columns=config.retrieval.top_k_columns,
                    max_candidate_tables=config.retrieval.max_candidate_tables,
                    score_threshold=config.retrieval.score_threshold,
                )
                self._retriever.build_index(force_rebuild=config.retrieval.force_rebuild)
                logger.info("[SkillBuilder] DualTowerRetriever ready")
            except Exception as e:
                logger.warning(f"[SkillBuilder] DualTowerRetriever init failed (will use full schema): {e}")
                self._retriever = None
        
        # Initialize SessionPlanManager for task tracking
        self._plan_manager = None
        try:
            from agent.session_plan import SessionPlanManager
            import os
            sessions_dir = os.path.join(config.output.report_dir, "sessions")
            self._plan_manager = SessionPlanManager(base_dir=sessions_dir)
            logger.info(f"[SkillBuilder] SessionPlanManager ready (dir={sessions_dir})")
        except Exception as e:
            logger.warning(f"[SkillBuilder] SessionPlanManager init failed: {e}")
            self._plan_manager = None
        
        # Initialize Skills
        logger.info("Initializing Skills...")
        simple_skill = SimpleQuerySkill(llm, tool_manager, db_manager)
        complex_skill = ComplexQuerySkill(llm, tool_manager, db_manager, retriever=self._retriever, plan_manager=self._plan_manager)
        analysis_skill = DataAnalysisSkill(llm, tool_manager, db_manager, config=config, plan_manager=self._plan_manager)
        
        # Register all skills — descriptions loaded from SKILL.md automatically
        self.registry = SkillRegistry()
        self.registry.register(simple_skill)
        self.registry.register(complex_skill)
        self.registry.register(analysis_skill)
        
        logger.info(f"Skills registered: {self.registry.list_skills()}")
    
    def _make_skill_node(self, skill) -> Callable:
        """Create a graph node callable for the given skill."""
        def skill_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"[Main] Executing skill: {skill.name}")
            result = skill.invoke(state)
            return {
                "messages": result.get("messages", []),
                "skill_result": result,
            }
        skill_node.__name__ = f"{skill.name}_node"
        return skill_node

    def build(self) -> StateGraph:
        """Build the main skill-based graph."""
        from typing_extensions import TypedDict
        from typing import Annotated
        
        # Define main graph state
        class MainGraphState(TypedDict):
            messages: Annotated[list, add_messages]
            query_type: str  # matched skill name
            skill_result: dict  # Result from selected skill
        
        logger.info("Building Skill-based main graph")
        
        graph = StateGraph(MainGraphState)
        
        # Add router node
        graph.add_node("query_router", self._query_router_node)
        
        # Dynamically add one node per registered skill
        skill_names = self.registry.list_skills()
        for name, skill in self.registry.get_all().items():
            graph.add_node(name, self._make_skill_node(skill))
        
        # Define routing flow
        graph.add_edge(START, "query_router")
        
        # Conditional routing: keys come from registry, not hardcoded
        graph.add_conditional_edges(
            "query_router",
            self._route_to_skill,
            {name: name for name in skill_names},
        )
        
        # All skills end at END
        for name in skill_names:
            graph.add_edge(name, END)
        
        # Compile with checkpointer if provided
        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()
    
    def _query_router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route query to appropriate Skill based on registry descriptions.
        
        Skill descriptions are loaded dynamically from SKILL.md via SkillRegistry,
        enabling progressive disclosure: only skill summaries are sent to LLM here;
        full tool details are only exposed inside each skill's subgraph.
        """
        logger.info("[Router] Analyzing query complexity")
        
        messages = state.get("messages", [])
        # Use the latest HumanMessage for multi-turn conversation support
        user_question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break
        if not user_question and messages:
            user_question = messages[0].content
        
        # Build classification prompt dynamically from registry (sourced from SKILL.md)
        skill_descriptions = self.registry.build_router_prompt()
        valid_names = self.registry.list_skills()
        
        system_prompt = f"""你是一个查询意图分类器。根据用户问题，从以下可用技能中选择最合适的一个。

{skill_descriptions}

规则：
- 只输出技能名称，不要输出任何其他内容
- 必须是以下名称之一：{valid_names}
"""
        
        classification_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户问题：{user_question}\n\n请输出技能名称。")
        ]
        
        response = self.llm.invoke(classification_messages)
        classification = response.content.strip().lower()
        
        # Validate — fallback to first registered skill if unknown
        if classification not in valid_names:
            fallback = valid_names[0]
            logger.warning(f"[Router] Unknown classification '{classification}', defaulting to '{fallback}'")
            classification = fallback
        
        logger.info(f"[Router] Query classified as: {classification}")
        
        new_message = AIMessage(content=f"Query Type: {classification.upper()}")
        
        return {
            "messages": [new_message],
            "query_type": classification,
        }
    
    def _route_to_skill(self, state: Dict[str, Any]) -> str:
        """Determine which skill to route to based on query_type."""
        query_type = state.get("query_type", self.registry.list_skills()[0])
        logger.info(f"[Router] Routing to skill: {query_type}")
        return query_type


def create_skill_based_graph(
    config: AgentConfig,
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver = None
) -> StateGraph:
    """
    Create the Skill-based SQL Agent graph.
    
    Args:
        config: Agent configuration
        llm: Language model instance (BaseChatModel or RunnableRetry wrapping one)
        checkpointer: Optional checkpointer for memory
        
    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating Skill-based SQL Agent graph")

    # Skills need BaseChatModel for bind_tools(); unwrap RunnableRetry if needed.
    # Correct retry pattern: base_llm.bind_tools(tools).with_retry(...), not the
    # other way around. We pass the base model to all components; each call site
    # that needs retry can call .with_retry() after bind_tools().
    base_llm = llm
    if not isinstance(llm, BaseChatModel) and hasattr(llm, "bound"):
        base_llm = llm.bound

    # Initialize managers
    db_manager = SQLDatabaseManager(config.database, security_config=config.security)
    tool_manager = SQLToolManager(db_manager, base_llm)
    
    # Build graph
    builder = SkillBasedGraphBuilder(
        config=config,
        llm=base_llm,
        db_manager=db_manager,
        tool_manager=tool_manager,
        checkpointer=checkpointer
    )
    
    graph = builder.build()
    
    logger.info("Skill-based SQL Agent graph created successfully")
    return graph

