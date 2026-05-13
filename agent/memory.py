"""对话记忆增强系统。

三层记忆架构:
  1. MessageFilter       — 过滤中间工具消息，只保留 Human + 最终AI回答对
  2. ConversationMemoryManager — 每 N 轮对旧对话做 LLM 摘要，存为记忆卡片
  3. SQLiteCardStore     — 持久化记忆卡片（向量 + 元数据），支持语义检索
                           Milvus 不可用时自动降级到 SQLite + numpy

记忆卡片策略:
  - 原始消息 → 过滤 → 分轮次 → 超出窗口的旧轮次 → LLM 摘要 → 卡片
  - 卡片绝不二次摘要（防止信息叠加损耗）
  - 检索时: embed(当前问题) → cosine 相似度 → top-k 卡片
  - 上下文组装: [相关卡片摘要] + [近 WINDOW_TURNS 轮完整] + [当前问题]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage

logger = logging.getLogger(__name__)


# ── 嵌入模型（复用 column_index 中的缓存） ───────────────────────────────────

def _get_encoder(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """返回 SentenceTransformer 编码器，失败时返回 None（降级模式）。"""
    try:
        from agent.column_index import _get_cached_encoder
        return _get_cached_encoder(model_name)
    except Exception as e:
        logger.warning(f"[Memory] Embedding model unavailable: {e}. Using keyword fallback.")
        return None


# ── 数据模型 ─────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """一个完整的对话轮次（Human 问 + AI 答）。"""
    turn_index: int           # 从 0 开始的轮次序号
    human: str                # 用户消息
    ai: str = ""              # AI 最终回答（空 = 当前轮未结束）


@dataclass
class MemoryCard:
    """一张记忆卡片，对应若干轮次的 LLM 摘要。"""
    card_id: str
    thread_id: str
    turn_start: int           # 覆盖的起始轮次（含）
    turn_end: int             # 覆盖的结束轮次（含）
    summary: str              # LLM 生成的摘要文本
    created_at: str           # ISO-8601 时间戳
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ConversationContext:
    """组装好的 LLM 上下文，供 ContextAssembler 格式化。"""
    memory_cards: List[MemoryCard] = field(default_factory=list)
    window_turns: List[ConversationTurn] = field(default_factory=list)
    session_plan_text: str = ""           # session_plan.md 的摘要


# ── 消息过滤器 ────────────────────────────────────────────────────────────────

class MessageFilter:
    """过滤掉中间工具调用消息，只保留 Human 消息和最终 AI 回答。

    保留规则:
      - 所有 HumanMessage
      - 没有 tool_calls 且有内容的 AIMessage（即最终自然语言回答）
    过滤规则:
      - 含 tool_calls 的 AIMessage（SQL 生成指令）
      - ToolMessage（SQL 执行结果）
      - 带 "Available tables:"、"Database schema:" 等前缀的 AIMessage（schema 缓存）
    """

    _NOISE_PREFIXES = (
        "Available tables:",
        "Database schema:",
        "Query Type:",
        "Error listing tables:",
        "Error retrieving schema:",
    )

    @classmethod
    def filter(cls, messages: List[AnyMessage]) -> List[AnyMessage]:
        result: List[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append(msg)
            elif isinstance(msg, AIMessage):
                # 跳过含工具调用的中间消息
                if getattr(msg, "tool_calls", None):
                    continue
                # 跳过噪声前缀的消息
                content = (msg.content or "").strip()
                if any(content.startswith(p) for p in cls._NOISE_PREFIXES):
                    continue
                if content:
                    result.append(msg)
        return result

    @classmethod
    def group_into_turns(cls, filtered: List[AnyMessage]) -> List[ConversationTurn]:
        """将过滤后的消息序列分组为轮次列表。"""
        turns: List[ConversationTurn] = []
        idx = 0
        while idx < len(filtered):
            msg = filtered[idx]
            if isinstance(msg, HumanMessage):
                turn = ConversationTurn(turn_index=len(turns), human=msg.content)
                idx += 1
                # 找紧跟的 AI 回答
                if idx < len(filtered) and isinstance(filtered[idx], AIMessage):
                    turn.ai = filtered[idx].content
                    idx += 1
                turns.append(turn)
            else:
                idx += 1  # 跳过孤立的 AI 消息
        return turns


# ── SQLite 卡片存储 ───────────────────────────────────────────────────────────

class SQLiteCardStore:
    """记忆卡片持久化存储（SQLite + numpy 向量相似度）。

    数据库表:
      memory_cards   — 卡片内容与嵌入向量
      memory_cursors — 每个 thread_id 已完成摘要的轮次游标
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._setup()

    def _setup(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_cards (
                    card_id     TEXT PRIMARY KEY,
                    thread_id   TEXT NOT NULL,
                    turn_start  INTEGER NOT NULL,
                    turn_end    INTEGER NOT NULL,
                    summary     TEXT NOT NULL,
                    embedding   BLOB,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread ON memory_cards(thread_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_cursors (
                    thread_id   TEXT PRIMARY KEY,
                    cursor      INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    def add(self, card: MemoryCard) -> None:
        emb_blob = card.embedding.astype(np.float32).tobytes() if card.embedding is not None else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_cards VALUES (?,?,?,?,?,?,?)",
                (card.card_id, card.thread_id, card.turn_start, card.turn_end,
                 card.summary, emb_blob, card.created_at)
            )
            conn.commit()
        logger.debug(f"[MemoryStore] Saved card {card.card_id} for thread {card.thread_id[:8]}")

    def get_cursor(self, thread_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT cursor FROM memory_cursors WHERE thread_id=?", (thread_id,)
            ).fetchone()
        return row[0] if row else 0

    def set_cursor(self, thread_id: str, cursor: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_cursors (thread_id, cursor) VALUES (?,?)",
                (thread_id, cursor)
            )
            conn.commit()

    def search(
        self,
        thread_id: str,
        query_embedding: Optional[np.ndarray],
        top_k: int = 3,
    ) -> List[MemoryCard]:
        """检索与 query_embedding 最相关的卡片（cosine 相似度）。

        若 query_embedding 为 None（降级模式），返回最新的 top_k 张卡片。
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT card_id,thread_id,turn_start,turn_end,summary,embedding,created_at "
                "FROM memory_cards WHERE thread_id=? ORDER BY turn_start ASC",
                (thread_id,)
            ).fetchall()

        if not rows:
            return []

        cards = [
            MemoryCard(
                card_id=r[0], thread_id=r[1], turn_start=r[2], turn_end=r[3],
                summary=r[4],
                embedding=np.frombuffer(r[5], dtype=np.float32) if r[5] else None,
                created_at=r[6],
            )
            for r in rows
        ]

        if query_embedding is None or not any(c.embedding is not None for c in cards):
            # 降级：返回最新的 top_k
            return cards[-top_k:]

        # 计算 cosine 相似度并排序
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scored: List[Tuple[float, MemoryCard]] = []
        for card in cards:
            if card.embedding is not None:
                v = card.embedding / (np.linalg.norm(card.embedding) + 1e-9)
                sim = float(np.dot(q, v))
                scored.append((sim, card))
            else:
                scored.append((0.0, card))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def has_cards(self, thread_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memory_cards WHERE thread_id=?", (thread_id,)
            ).fetchone()[0]
        return count > 0


# ── 摘要 Prompt ────────────────────────────────────────────────────────────────

_SUMMARY_SYSTEM_PROMPT = """[Role & Policies]
你是一个对话摘要助手，为 Text-to-SQL 系统生成历史记忆卡片。
摘要要精准、压缩，保留核心信息，不添加推断内容。

[Task]
将提供的对话轮次压缩成简洁的中文摘要（不超过200字）。

[Environment]
（无）

[Evidence]
（无）

[Context]
（对话轮次由调用方以 HumanMessage 传入）

[Output]
重点保留以下信息：
1. 用户查询了什么数据（表名、条件、目的）
2. 得到了什么关键结论或数字
3. 有哪些重要的上下文（如用户偏好、特定业务逻辑）

只输出摘要内容，不要输出任何前缀或解释。"""


# ── 对话记忆管理器 ─────────────────────────────────────────────────────────────

class ConversationMemoryManager:
    """统一的对话记忆管理入口。

    参数:
        llm: 用于生成摘要的语言模型
        db_path: SQLite 记忆卡片数据库路径
        window_turns: 保留在完整滑动窗口中的轮次数（默认 3）
        summary_every_n: 累积多少旧轮次后触发摘要（默认 5）
        top_k_cards: 检索时返回最相关的卡片数（默认 3）
        embedding_model: SentenceTransformer 模型名称
    """

    def __init__(
        self,
        llm: BaseChatModel,
        db_path: str = "memory_cards.db",
        window_turns: int = 5,
        summary_every_n: int = 5,
        top_k_cards: int = 3,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        max_window_tokens: int = 6000,
        dedup_threshold: float = 0.85,
    ):
        self.llm = llm
        self.store = SQLiteCardStore(db_path)
        self.window_turns = window_turns
        self.summary_every_n = summary_every_n
        self.top_k_cards = top_k_cards
        self._encoder = _get_encoder(embedding_model)
        self.max_window_tokens = max_window_tokens
        self.dedup_threshold = dedup_threshold

    # ── 编码 ──────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> Optional[np.ndarray]:
        if self._encoder is None:
            return None
        try:
            vec = self._encoder.encode(text, convert_to_numpy=True)
            return vec.astype(np.float32)
        except Exception as e:
            logger.warning(f"[Memory] Encode failed: {e}")
            return None

    # ── token 估算（粗粒度：4字符≈1token） ───────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _window_token_count(self, turns: List[ConversationTurn]) -> int:
        total = 0
        for t in turns:
            total += self._estimate_tokens(t.human)
            if t.ai:
                total += self._estimate_tokens(t.ai)
        return total

    # ── 稀疏化保留：确保第一轮始终锚定在窗口 ────────────────────────────────

    @staticmethod
    def _anchor_first_turn(
        window: List[ConversationTurn],
        complete_turns: List[ConversationTurn],
    ) -> List[ConversationTurn]:
        """若 T1 不在当前窗口内，将其插入到窗口最前端（Anchor 策略）。"""
        if not complete_turns or not window:
            return window
        anchor = complete_turns[0]
        if anchor is window[0]:
            return window
        return [anchor] + window

    # ── 去重压缩：在 format_history_messages 中跳过语义重复的旧轮次 ──────────

    def _dedup_window_indices(
        self, window: List[ConversationTurn]
    ) -> set:
        """返回应被跳过输出的轮次索引集合（保留较新，跳过较旧的相似轮次）。

        两两比较窗口内问题的语义相似度，超过 dedup_threshold 时标记较旧的
        那条为「跳过」。原始 window 列表不被修改，以确保摘要计算不受影响。
        """
        if self._encoder is None or len(window) < 2:
            return set()

        embeddings: List[Optional[np.ndarray]] = [
            self.encode(t.human) for t in window
        ]
        to_skip: set = set()
        for i in range(len(window)):
            if i in to_skip or embeddings[i] is None:
                continue
            for j in range(i + 1, len(window)):
                if j in to_skip or embeddings[j] is None:
                    continue
                # 余弦相似度
                a, b = embeddings[i], embeddings[j]
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    continue
                sim = float(np.dot(a, b) / (norm_a * norm_b))
                if sim >= self.dedup_threshold:
                    to_skip.add(i)  # 旧的（i < j）跳过，保留较新的 j
                    logger.debug(
                        f"[Memory] Dedup: skip turn {i} (sim={sim:.2f} with turn {j})"
                    )
                    break  # i 已标记，不必再比
        return to_skip

    # ── 摘要生成 ──────────────────────────────────────────────────────────────

    def _summarize_turns(self, turns: List[ConversationTurn]) -> str:
        """对一批轮次调用 LLM 生成摘要文本。"""
        lines: List[str] = []
        for t in turns:
            lines.append(f"用户: {t.human}")
            if t.ai:
                ai_snippet = t.ai[:300] + ("..." if len(t.ai) > 300 else "")
                lines.append(f"助手: {ai_snippet}")
        conversation_text = "\n".join(lines)
        try:
            resp = self.llm.invoke([
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=conversation_text),
            ])
            return (resp.content or "").strip()
        except Exception as e:
            logger.warning(f"[Memory] LLM summarize failed: {e}")
            # 降级：直接截取前几轮作为摘要
            return conversation_text[:400]

    # ── 主入口：获取对话上下文 ─────────────────────────────────────────────────

    def get_context(
        self,
        thread_id: str,
        messages: List[AnyMessage],
        session_plan_text: str = "",
    ) -> ConversationContext:
        """根据消息历史和当前问题，组装 LLM 上下文。

        1. 过滤中间消息 → 分轮次
        2. 将超出窗口且未摘要的旧轮次批量摘要 → 存卡片
        3. 方案A：把不够触发摘要的「零头」旧轮次临时并入窗口（不丢失）
        4. Token 超限时从窗口最前端弹出旧轮次并追加摘要（token 限流）
        5. 检索相关记忆卡片
        6. 返回 ConversationContext
        """
        filtered = MessageFilter.filter(messages)
        all_turns = MessageFilter.group_into_turns(filtered)

        # 完整轮次（最后一轮可能没有 AI 回答，排除在窗口外计算之外）
        complete_turns = [t for t in all_turns if t.ai]
        current_turn_human = all_turns[-1].human if all_turns and not all_turns[-1].ai else ""

        # 滑动窗口（最近 window_turns 个完整轮次）
        window = complete_turns[-self.window_turns:]
        old_turns = complete_turns[:-self.window_turns] if len(complete_turns) > self.window_turns else []

        # ── 触发摘要（整批） ──────────────────────────────────────────────────
        cursor = self.store.get_cursor(thread_id)
        unsummarized = old_turns[cursor:]

        if len(unsummarized) >= self.summary_every_n:
            logger.info(f"[Memory] Summarizing {len(unsummarized)} turns for thread {thread_id[:8]}")
            while len(unsummarized) >= self.summary_every_n:
                batch = unsummarized[:self.summary_every_n]
                summary_text = self._summarize_turns(batch)
                embedding = self.encode(summary_text)

                card = MemoryCard(
                    card_id=str(uuid.uuid4()),
                    thread_id=thread_id,
                    turn_start=cursor,
                    turn_end=cursor + len(batch) - 1,
                    summary=summary_text,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    embedding=embedding,
                )
                self.store.add(card)
                logger.info(
                    f"[Memory] Card saved: turns {card.turn_start}-{card.turn_end}, "
                    f"len={len(summary_text)}"
                )
                cursor += len(batch)
                unsummarized = unsummarized[self.summary_every_n:]

            self.store.set_cursor(thread_id, cursor)

        # ── 方案 A：零头并入窗口，确保不丢失 ─────────────────────────────────
        # unsummarized 此时是不足 summary_every_n 的零头（已更新 cursor 后剩余）
        remainder = old_turns[cursor:]
        if remainder:
            logger.debug(f"[Memory] Merging {len(remainder)} remainder turns into window")
            window = remainder + window

        # ── Token 限流：收集需要弹出的最旧轮次，一次性批量摘要为 1 张卡片 ───────
        # 策略：从最旧轮次开始向最新累加，直到移除这些轮次后 token 降至限制内，
        # 再将收集到的轮次一起摘要，避免多次 LLM 调用，也能处理大 token 轮次在中间的情况。
        if len(window) > 1 and self._window_token_count(window) > self.max_window_tokens:
            to_evict: List[ConversationTurn] = []
            while len(window) > 1 and self._window_token_count(window) > self.max_window_tokens:
                to_evict.append(window.pop(0))

            if to_evict:
                logger.info(
                    f"[Memory] Token overflow: evicting {len(to_evict)} turns "
                    f"(T{to_evict[0].turn_index}~T{to_evict[-1].turn_index})"
                )
                summary_text = self._summarize_turns(to_evict)
                embedding = self.encode(summary_text)
                card = MemoryCard(
                    card_id=str(uuid.uuid4()),
                    thread_id=thread_id,
                    turn_start=to_evict[0].turn_index,
                    turn_end=to_evict[-1].turn_index,
                    summary=summary_text,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    embedding=embedding,
                )
                self.store.add(card)
                cursor = to_evict[-1].turn_index + 1
                self.store.set_cursor(thread_id, cursor)
                logger.info(
                    f"[Memory] Token-overflow card saved (turns {card.turn_start}~{card.turn_end}), "
                    f"window now {len(window)} turns"
                )

        # ── 稀疏化保留：将 T1 固定在窗口最前端 ───────────────────────────────
        window = self._anchor_first_turn(window, complete_turns)

        # ── 检索相关记忆卡片 ──────────────────────────────────────────────────
        relevant_cards: List[MemoryCard] = []
        if self.store.has_cards(thread_id) and current_turn_human:
            query_emb = self.encode(current_turn_human)
            relevant_cards = self.store.search(thread_id, query_emb, top_k=self.top_k_cards)
            logger.debug(f"[Memory] Retrieved {len(relevant_cards)} memory cards")

        return ConversationContext(
            memory_cards=relevant_cards,
            window_turns=window,
            session_plan_text=session_plan_text,
        )

    # ── 上下文格式化 ──────────────────────────────────────────────────────────

    def format_history_messages(self, context: ConversationContext) -> List[AnyMessage]:
        """将 ConversationContext 转为可注入 LLM 的消息列表。

        顺序: [记忆卡片摘要 SystemMessage] + [近N轮 Human/AI 消息]
        （不包含当前问题，当前问题由调用方拼接）

        去重压缩：语义相似的旧轮次在此跳过输出，但不从 window 物理删除，
        以确保摘要游标计算不受影响。
        """
        result: List[AnyMessage] = []

        # 记忆卡片
        if context.memory_cards:
            card_lines = []
            for i, card in enumerate(context.memory_cards):
                card_lines.append(f"[记忆卡片 {i+1}] (轮次 {card.turn_start+1}~{card.turn_end+1})\n{card.summary}")
            result.append(SystemMessage(
                content="## 相关历史记忆（来自更早的对话摘要）\n\n" + "\n\n".join(card_lines)
            ))

        # 任务进度摘要（来自 session_plan）
        if context.session_plan_text:
            result.append(SystemMessage(
                content=f"## 当前任务进度\n{context.session_plan_text}"
            ))

        # 去重：计算应跳过输出的轮次索引（不影响 window 本身）
        skip_indices = self._dedup_window_indices(context.window_turns)
        if skip_indices:
            logger.debug(f"[Memory] Dedup skipping {len(skip_indices)} turns in output")

        # 近 N 轮消息（跳过语义重复的旧轮次）
        for idx, turn in enumerate(context.window_turns):
            if idx in skip_indices:
                continue
            result.append(HumanMessage(content=turn.human))
            if turn.ai:
                result.append(AIMessage(content=turn.ai))

        return result
