"""
Skill-based Main Graph Builder

Integrates three Skills (Simple Query, Complex Query, Data Analysis)
with two-level intelligent routing:
  L1 intent_router  — general_chat vs db_query  (lightweight, no DB context)
  L2 skill_router   — simple_query / complex_query / data_analysis (DB path only)

Memory System:
  ConversationMemoryManager is initialized here and injected into all LLM call
  sites. It provides:
    - MessageFilter: strips intermediate tool-call messages from history
    - Sliding window (last WINDOW_TURNS complete turns) passed in full
    - Summary memory cards: older turns are summarized by LLM → stored in SQLite
    - Vector retrieval: relevant cards retrieved for each new question
"""

import logging
from typing import Any, Dict, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.config import AgentConfig
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from agent.skills.registry import SkillRegistry

# Note: Skills are imported lazily in __init__ to avoid circular imports

logger = logging.getLogger(__name__)

# L1 classifier prompt — includes conversation context for follow-up awareness
_INTENT_SYSTEM_PROMPT = """[Role & Policies]
你是一个意图分类器，专门判断用户问题是否需要查询数据库。
规则：只输出 db_query 或 general_chat，不输出任何其他内容。

[Task]
判断当前用户消息的意图类别：
- db_query：涉及查询数据、统计分析、或对历史查询结果的追问（如"那第一名是谁"、"它的利润率呢"）
- general_chat：闲聊、打招呼、问天气、问你是谁等与数据库无关的问题

[Environment]
（无）

[Evidence]
（无）

[Context]
结合对话历史进行判断。如果历史中已有数据库查询的上下文，追问类问题应归为 db_query。

[Output]
只输出 db_query 或 general_chat，不附加任何解释。"""

# general_chat node system prompt
_GENERAL_CHAT_SYSTEM_PROMPT = """[Role & Policies]
你是一个 Text-to-SQL 智能助手，专门帮助用户查询和分析数据库。
对非数据库问题友好直接回答；不猜测、不编造数据。

[Task]
用户当前的问题不需要查询数据库，直接回答即可。
如果用户想查询数据，告知可用自然语言描述需求，你会帮他们转换成 SQL 查询。

[Environment]
（无）

[Evidence]
（无）

[Context]
（由记忆系统注入）

[Output]
使用中文回答，语气友好简洁。"""

# format_answer node system prompt — wraps raw SQL results into natural language
_FORMAT_ANSWER_SYSTEM_PROMPT = """[Role & Policies]
你是一个 Text-to-SQL 智能助手，负责将原始 SQL 查询结果转换为自然语言回答。
不重复 SQL 语句，不编造数据，只基于提供的查询结果作答。

[Task]
根据用户问题和 SQL 查询结果，生成简洁友好的中文回答。

[Environment]
（无）

[Evidence]
（用户问题和 SQL 查询结果由调用方注入到 HumanMessage 中）

[Context]
（无）

[Output]
- 直接回答用户问题，不重复 SQL 语句
- 如果结果是数字，说明含义（如"共有 1006 个用户"）
- 如果结果是列表，摘要展示（不超过 10 条详细列出，更多则说明总数）
- 使用中文回答"""


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

        # Initialize ConversationMemoryManager
        self._memory: Any = None
        if config.memory.enabled:
            try:
                import os
                from agent.memory import ConversationMemoryManager
                self._memory = ConversationMemoryManager(
                    llm=llm,
                    db_path=config.memory.db_path,
                    window_turns=config.memory.window_turns,
                    summary_every_n=config.memory.summary_every_n,
                    top_k_cards=config.memory.top_k_cards,
                    embedding_model=config.memory.embedding_model,
                    max_window_tokens=config.memory.max_window_tokens,
                    dedup_threshold=config.memory.dedup_threshold,
                )
                logger.info("[SkillBuilder] ConversationMemoryManager ready")
            except Exception as e:
                logger.warning(f"[SkillBuilder] MemoryManager init failed (will use raw history): {e}")
                self._memory = None
        
        # Initialize Skills
        logger.info("Initializing Skills...")
        confirm_enabled = config.sql_confirm_enabled
        simple_skill = SimpleQuerySkill(llm, tool_manager, db_manager, confirm_enabled=confirm_enabled)
        complex_skill = ComplexQuerySkill(llm, tool_manager, db_manager, retriever=self._retriever, plan_manager=self._plan_manager, confirm_enabled=confirm_enabled)
        analysis_skill = DataAnalysisSkill(llm, tool_manager, db_manager, config=config, plan_manager=self._plan_manager, confirm_enabled=confirm_enabled)
        
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

        # Final answer formatter — runs after every skill to produce natural language reply
        graph.add_node("format_answer", self._format_answer_node)
        
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
        # All skills → format_answer → END
        for name in skill_names:
            graph.add_edge(name, "format_answer")
        graph.add_edge("format_answer", END)
        
        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()

    # ── L1: Intent routing ────────────────────────────────────────────────────

    @staticmethod
    def _extract_thread_id(config: Any) -> str:
        """从 LangGraph RunnableConfig 中安全提取 thread_id。"""
        if config is None:
            return ""
        if hasattr(config, "get"):
            return config.get("configurable", {}).get("thread_id", "")
        return ""

    def _intent_router_node(self, state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        """L1 router: decide general_chat vs db_query.

        Uses ConversationMemoryManager (if enabled) to:
          - Filter intermediate tool messages from history
          - Inject relevant memory cards from older turns
          - Pass only clean sliding window history to classifier
        Falls back to raw last-6-messages logic when memory is disabled.
        """
        logger.info("[L1 Router] Classifying intent")

        messages = state.get("messages", [])
        current_question = self._latest_human_message(state)
        thread_id = self._extract_thread_id(config)

        classification_messages = [SystemMessage(content=_INTENT_SYSTEM_PROMPT)]

        if self._memory is not None and thread_id:
            # Use memory manager: get filtered + summarized context
            ctx = self._memory.get_context(thread_id, messages)
            history_msgs = self._memory.format_history_messages(ctx)
            # Build plain-text history for classifier (exclude current question)
            history_lines = []
            for msg in history_msgs:
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"用户: {msg.content[:200]}")
                elif isinstance(msg, AIMessage):
                    content = msg.content[:200] + ("..." if len(msg.content) > 200 else "")
                    history_lines.append(f"助手: {content}")
                elif isinstance(msg, SystemMessage) and "记忆卡片" in msg.content:
                    prefix = "## 相关历史记忆（来自更早的对话摘要）\n\n"
                    history_lines.append(f"[历史摘要] {msg.content[len(prefix):][:300]}")
            if history_lines:
                classification_messages.append(
                    SystemMessage(content=f"【对话历史】\n" + "\n".join(history_lines))
                )
        else:
            # Fallback: raw last-6-messages (original behavior)
            recent = messages[-6:] if len(messages) > 6 else messages
            history_lines = []
            for msg in recent:
                if not hasattr(msg, "type"):
                    continue
                if msg.type == "human":
                    history_lines.append(f"用户: {msg.content}")
                elif msg.type == "ai":
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    history_lines.append(f"助手: {content}")
            if history_lines:
                history_text = "\n".join(history_lines[:-1])
                if history_text:
                    classification_messages.append(
                        SystemMessage(content=f"【对话历史】\n{history_text}")
                    )

        classification_messages.append(HumanMessage(content=current_question))

        response = self.llm.invoke(classification_messages)
        intent = response.content.strip().lower()

        if intent not in ("general_chat", "db_query"):
            logger.warning(f"[L1 Router] Unknown intent '{intent}', defaulting to db_query")
            intent = "db_query"

        logger.info(f"[L1 Router] Intent → {intent}")
        return {"query_intent": intent}

    def _route_intent(self, state: Dict[str, Any]) -> str:
        return state.get("query_intent", "db_query")

    async def _general_chat_node(self, state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        """Direct LLM reply for non-database questions.

        Uses memory context (filtered window + memory cards) when available.
        Falls back to passing full raw history when memory is disabled.
        """
        logger.info("[General Chat] Responding without DB")
        messages = state.get("messages", [])
        thread_id = self._extract_thread_id(config)

        chat_messages = [SystemMessage(content=_GENERAL_CHAT_SYSTEM_PROMPT)]

        if self._memory is not None and thread_id:
            ctx = self._memory.get_context(thread_id, messages)
            chat_messages += self._memory.format_history_messages(ctx)
            chat_messages.append(HumanMessage(content=self._latest_human_message(state)))
        else:
            chat_messages += messages

        response = await self.llm.ainvoke(chat_messages)
        return {"messages": [response]}

    async def _format_answer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format the raw SQL query result into a natural language answer."""
        from langchain_core.messages import ToolMessage
        messages = state.get("messages", [])
        # Collect recent context: last human message + any tool results
        user_question = self._latest_human_message(state)
        tool_results = [
            m.content for m in messages
            if isinstance(m, ToolMessage) and m.content
        ]
        if not tool_results:
            # No SQL result — nothing to format, skip
            return {}
        result_text = "\n".join(tool_results[-3:])  # at most last 3 tool outputs
        format_messages = [
            SystemMessage(content=_FORMAT_ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{user_question}\n\nSQL 查询结果：\n{result_text}"),
        ]
        response = await self.llm.ainvoke(format_messages)
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

