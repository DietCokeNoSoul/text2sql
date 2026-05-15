import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, ToolMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.simple_query.skill import SimpleQuerySkill


def test_validate_result_uses_latest_human_message():
    skill = SimpleQuerySkill.__new__(SimpleQuerySkill)
    skill.llm = MagicMock()
    skill.llm.invoke.return_value = SimpleNamespace(content="VALID")

    state = {
        "messages": [
            HumanMessage(content="你好"),
            HumanMessage(content="有多少张优惠券"),
            ToolMessage(content="[(42,)]", tool_call_id="call-1", name="sql_db_query"),
        ],
        "last_sql": "SELECT COUNT(*) AS voucher_count FROM tb_voucher LIMIT 1000",
        "retry_count": 0,
    }

    result = skill._validate_result(state)

    assert result == {"validation_feedback": ""}
    prompt = skill.llm.invoke.call_args.args[0][1]["content"]
    assert "用户问题：有多少张优惠券" in prompt
    assert "用户问题：你好" not in prompt