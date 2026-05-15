import os
import sys
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.skill_graph_builder import SkillBasedGraphBuilder


def test_format_answer_returns_raw_sql_result_when_llm_timeout():
    builder = SkillBasedGraphBuilder.__new__(SkillBasedGraphBuilder)
    builder.llm = AsyncMock()
    builder.llm.ainvoke.side_effect = TimeoutError("Connection timed out")

    state = {
        "messages": [
            HumanMessage(content="查询用户数量"),
            ToolMessage(content="[(1006,)]", tool_call_id="call-1", name="sql_db_query"),
        ],
        "constraints": [],
    }

    result = __import__("asyncio").run(builder._format_answer_node(state))

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "[(1006,)]"
