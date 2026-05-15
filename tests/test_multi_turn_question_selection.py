import os
import sys

from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.complex_query.skill import ComplexQuerySkill
from skills.data_analysis.skill import DataAnalysisSkill


def test_complex_query_uses_latest_human_message():
    skill = ComplexQuerySkill.__new__(ComplexQuerySkill)

    state = {
        "messages": [
            HumanMessage(content="你好"),
            HumanMessage(content="统计每家店铺拥有的优惠券数量"),
        ]
    }

    assert skill._get_user_question(state) == "统计每家店铺拥有的优惠券数量"


def test_data_analysis_uses_latest_human_message():
    messages = [
        HumanMessage(content="你好"),
        HumanMessage(content="分析用户活跃度趋势"),
    ]

    assert DataAnalysisSkill._latest_human_message(messages) == "分析用户活跃度趋势"