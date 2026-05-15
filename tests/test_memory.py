import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.memory import ConversationMemoryManager


@pytest.fixture
def memory_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.memory._get_encoder", lambda _model: None)
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.invoke.return_value = SimpleNamespace(content="历史摘要")
    return ConversationMemoryManager(
        llm=mock_llm,
        db_path=str(tmp_path / "memory_cards.db"),
        window_turns=2,
        summary_every_n=2,
    )


def test_get_context_does_not_anchor_first_turn_after_summarization(memory_manager):
    messages = [
        HumanMessage(content="第一轮问题"),
        AIMessage(content="第一轮回答"),
        HumanMessage(content="第二轮问题"),
        AIMessage(content="第二轮回答"),
        HumanMessage(content="第三轮问题"),
        AIMessage(content="第三轮回答"),
        HumanMessage(content="第四轮问题"),
        AIMessage(content="第四轮回答"),
        HumanMessage(content="当前问题"),
    ]

    context = memory_manager.get_context("thread-1", messages)

    assert [turn.human for turn in context.window_turns] == ["第三轮问题", "第四轮问题"]
    assert len(context.memory_cards) == 1

    history_messages = memory_manager.format_history_messages(context)
    history_human_messages = [msg.content for msg in history_messages if isinstance(msg, HumanMessage)]
    assert history_human_messages == ["第三轮问题", "第四轮问题"]