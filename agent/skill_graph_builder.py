"""
Skill-based Main Graph Builder

Integrates three Skills (Simple Query, Complex Query, Data Analysis)
with two-level intelligent routing:
  L1 intent_router  — general_chat vs db_query  (lightweight, no DB context)
  L2 skill_router   — simple_query / complex_query / data_analysis (DB path only)
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

# L1 classifier prompt — includes conversation context for follow-up awareness
_INTENT_SYSTEM_PROMPT = """你是一个意图分类器，判断用户的问题是否需要查询数据库。

回答规则：
- 如果问题涉及查数据、统计、分析数据库中的信息 → 输出 db_query
- 如果问题是对历史查询结果的追问（如"那第一名是谁"、"它的利润率呢"）→ 输出 db_query
- 如果问题是闲聊、打招呼、问天气、问你是谁、或任何与数据库无关的问题 → 输出 general_chat
- 只输出 db_query 或 general_chat，不输出任何其他内容

注意：结合对话历史判断。如果历史中已有数据库查询的上下文，追问类问题应归为 db_query。"""

# general_chat node system prompt
_GENERAL_CHAT_SYSTEM_PROMPT = """你是一个 Text-to-SQL 智能助手，专门帮助用户查询和分析数据库。

用户当前的问题不需要查询数据库，请友好地直接回答。
如果用户想查询数据，可以告诉他们用自然语言描述需求，你会帮他们转换成 SQL 查询。"""


class SkillBasedGraphBuilder:
    """
    Build main graph with two-level routing.

    Flow:
        user_input
          → intent_router (L1: general_chat | db_query)
              ├─ general_chat  → general_chat_node → END
              └─ db_query      → skill_router (L2: simple_query | complex_query | data_analysis)
                                    └─ <skill_node> → END
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
        """Build the two-level routed main graph."""
        from typing_extensions import TypedDict
        from typing import Annotated
        
        class MainGraphState(TypedDict):
            messages: Annotated[list, add_messages]
            query_intent: str   # "general_chat" | "db_query"
            query_type: str     # skill name (set only when query_intent == "db_query")
            skill_result: dict  # result from selected skill
        
        logger.info("Building Skill-based main graph")
        
        graph = StateGraph(MainGraphState)
        
        # ── L1: Intent router ─────────────────────────────────────────────────
        graph.add_node("intent_router", self._intent_router_node)
        graph.add_node("general_chat", self._general_chat_node)

        # ── L2: Skill router ──────────────────────────────────────────────────
        graph.add_node("skill_router", self._skill_router_node)
        
        # Dynamically add one node per registered skill
        skill_names = self.registry.list_skills()
        for name, skill in self.registry.get_all().items():
            graph.add_node(name, self._make_skill_node(skill))
        
        # ── Edges ─────────────────────────────────────────────────────────────
        graph.add_edge(START, "intent_router")

        # L1 conditional: general_chat or db_query
        graph.add_conditional_edges(
            "intent_router",
            self._route_intent,
            {"general_chat": "general_chat", "db_query": "skill_router"},
        )
        graph.add_edge("general_chat", END)

        # L2 conditional: which skill
        graph.add_conditional_edges(
            "skill_router",
            self._route_to_skill,
            {name: name for name in skill_names},
        )
        for name in skill_names:
            graph.add_edge(name, END)
        
        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()

    # ── L1: Intent routing ────────────────────────────────────────────────────

    def _intent_router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """L1 router: decide general_chat vs db_query.

        Passes recent conversation history (last 6 messages) so the LLM can
        correctly handle follow-up questions that reference previous DB results.
        """
        logger.info("[L1 Router] Classifying intent")

        messages = state.get("messages", [])
        # Take last 6 messages as context window (3 turns), trim to save tokens
        recent = messages[-6:] if len(messages) > 6 else messages

        # Build history as plain text for the classifier
        history_lines = []
        for msg in recent:
            if not hasattr(msg, "type"):
                continue
            if msg.type == "human":
                history_lines.append(f"用户: {msg.content}")
            elif msg.type == "ai":
                # Truncate long AI replies to save tokens
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                history_lines.append(f"助手: {content}")

        classification_messages = [SystemMessage(content=_INTENT_SYSTEM_PROMPT)]
        if history_lines:
            history_text = "\n".join(history_lines[:-1])  # all but the last user message
            if history_text:
                classification_messages.append(
                    SystemMessage(content=f"【对话历史】\n{history_text}")
                )
        classification_messages.append(HumanMessage(content=self._latest_human_message(state)))

        response = self.llm.invoke(classification_messages)
        intent = response.content.strip().lower()

        if intent not in ("general_chat", "db_query"):
            logger.warning(f"[L1 Router] Unknown intent '{intent}', defaulting to db_query")
            intent = "db_query"

        logger.info(f"[L1 Router] Intent → {intent}")
        return {"query_intent": intent}

    def _route_intent(self, state: Dict[str, Any]) -> str:
        return state.get("query_intent", "db_query")

    def _general_chat_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Direct LLM reply for non-database questions."""
        logger.info("[General Chat] Responding without DB")
        user_question = self._latest_human_message(state)
        messages = [
            SystemMessage(content=_GENERAL_CHAT_SYSTEM_PROMPT),
            HumanMessage(content=user_question),
        ]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    # ── L2: Skill routing ─────────────────────────────────────────────────────

    def _skill_router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        L2 router: classify db_query into simple_query / complex_query / data_analysis.

        Skill descriptions loaded dynamically from SkillRegistry (sourced from SKILL.md),
        enabling progressive disclosure.
        """
        logger.info("[L2 Router] Classifying query complexity")

        user_question = self._latest_human_message(state)
        skill_descriptions = self.registry.build_router_prompt()
        valid_names = self.registry.list_skills()

        system_prompt = f"""你是一个数据库查询分类器。根据用户问题，从以下可用技能中选择最合适的一个。

{skill_descriptions}

规则：
- 只输出技能名称，不要输出任何其他内容
- 必须是以下名称之一：{valid_names}
"""
        classification_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户问题：{user_question}\n\n请输出技能名称。"),
        ]
        response = self.llm.invoke(classification_messages)
        classification = response.content.strip().lower()

        if classification not in valid_names:
            fallback = valid_names[0]
            logger.warning(f"[L2 Router] Unknown skill '{classification}', defaulting to '{fallback}'")
            classification = fallback

        logger.info(f"[L2 Router] Skill → {classification}")
        new_message = AIMessage(content=f"Query Type: {classification.upper()}")
        return {"messages": [new_message], "query_type": classification}

    def _route_to_skill(self, state: Dict[str, Any]) -> str:
        """Determine which skill to route to based on query_type."""
        query_type = state.get("query_type", self.registry.list_skills()[0])
        logger.info(f"[L2 Router] Routing to skill: {query_type}")
        return query_type

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _latest_human_message(state: Dict[str, Any]) -> str:
        """Extract the most recent HumanMessage content from state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
        return messages[0].content if messages else ""


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
    db_manager = SQLDatabaseManager(
        config.database,
        security_config=config.security,
        schema_cache_config=config.schema_cache,
    )
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

