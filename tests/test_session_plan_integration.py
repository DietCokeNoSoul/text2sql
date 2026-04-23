"""
Integration test for SessionPlanManager wired into ComplexQuerySkill.

Uses real SQLite (Chinook.db) + Mock LLM (no API key needed).
Verifies that plan.md / plan.json are written to report/sessions/{task_id}/.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from typing import Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ───────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "Chinook.db")
DB_URI = f"sqlite:///{os.path.abspath(DB_PATH)}"


def _make_ai_response(content: str):
    """Wrap text in a LangChain AIMessage-compatible mock."""
    from langchain_core.messages import AIMessage
    return AIMessage(content=content)


def _mock_llm_for_complex_query():
    """
    Returns a mock LLM that answers the ComplexQuerySkill planning prompt
    with a 2-step JSON plan, and the execution prompt with SQL results.
    """
    llm = MagicMock()

    plan_json = json.dumps({
        "steps": [
            {
                "step_id": 1,
                "description": "Get all Genre names and IDs",
                "query": "SELECT GenreId, Name FROM Genre ORDER BY Name LIMIT 5",
                "depends_on": []
            },
            {
                "step_id": 2,
                "description": "Count tracks per genre",
                "query": "SELECT g.Name, COUNT(t.TrackId) AS TrackCount "
                         "FROM Genre g JOIN Track t ON g.GenreId = t.GenreId "
                         "GROUP BY g.GenreId ORDER BY TrackCount DESC LIMIT 5",
                "depends_on": []
            }
        ]
    })

    # First invoke → plan response; subsequent → aggregate answer
    llm.invoke.side_effect = [
        _make_ai_response(plan_json),          # _plan_node
        _make_ai_response("Here is the summary of genres and their track counts."),  # _aggregate_node
    ]
    return llm


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sessions_dir(tmp_path):
    return str(tmp_path / "report" / "sessions")


@pytest.fixture
def plan_manager(sessions_dir):
    from agent.session_plan import SessionPlanManager
    return SessionPlanManager(base_dir=sessions_dir)


@pytest.fixture
def db_and_tools():
    """Real SQLite DB manager + tool manager pointing at Chinook.db."""
    from unittest.mock import MagicMock
    from langchain_core.language_models import BaseChatModel
    from agent.config import AgentConfig, DatabaseConfig, LLMConfig, OutputConfig, SecurityConfig, RetrievalConfig
    from agent.database import SQLDatabaseManager
    from agent.tools import SQLToolManager

    config = AgentConfig(
        database=DatabaseConfig(uri=DB_URI, max_query_results=10),
        llm=LLMConfig(provider="mock", model="mock", api_key="mock"),
        output=OutputConfig(report_dir="report"),
        security=SecurityConfig(),
        retrieval=RetrievalConfig(enabled=False),
    )
    db_manager = SQLDatabaseManager(config.database)
    # SQLToolManager requires a real BaseChatModel instance
    mock_llm = MagicMock(spec=BaseChatModel)
    tool_manager = SQLToolManager(db_manager, mock_llm)
    return db_manager, tool_manager


# ── Test: ComplexQuerySkill writes plan files ──────────────────────────────────

def test_complex_skill_creates_plan_files(db_and_tools, plan_manager, sessions_dir):
    """
    Run ComplexQuerySkill with a Mock LLM and a real SQLite DB.
    Verify:
      1. plan.json and plan.md are written under sessions_dir/{task_id}/
      2. plan.json has correct task structure (steps, status)
      3. plan.md contains the question title
      4. Steps that were executed show 'done' status
    """
    from langchain_core.messages import HumanMessage
    from skills.complex_query.skill import ComplexQuerySkill
    from agent.tools import SQLToolManager

    db_manager, tool_manager = db_and_tools
    llm = _mock_llm_for_complex_query()

    skill = ComplexQuerySkill(
        llm=llm,
        tool_manager=tool_manager,
        db_manager=db_manager,
        retriever=None,
        plan_manager=plan_manager,
    )

    question = "Which music genres have the most tracks?"
    result = skill.graph.invoke({
        "messages": [HumanMessage(content=question)],
        "tables": [],
        "table_schema": "",
        "query_plan": [],
        "step_results": {},
        "plan_completed": False,
        "retrieval_stats": {},
        "task_id": "",
    })

    # ── 1. task_id was assigned ───────────────────────────────────────────────
    task_id = result.get("task_id", "")
    assert task_id, "task_id should be non-empty after skill run"
    assert len(task_id) == 12

    # ── 2. Files exist on disk ────────────────────────────────────────────────
    task_dir = os.path.join(sessions_dir, task_id)
    json_path = os.path.join(task_dir, "plan.json")
    md_path = os.path.join(task_dir, "plan.md")

    assert os.path.exists(json_path), f"plan.json not found at {json_path}"
    assert os.path.exists(md_path),   f"plan.md not found at {md_path}"

    # ── 3. plan.json structure ────────────────────────────────────────────────
    data = json.loads(open(json_path, encoding="utf-8").read())
    assert data["task_id"] == task_id
    assert data["skill"] == "complex_query"
    assert len(data["steps"]) == 2

    # ── 4. Steps marked done ──────────────────────────────────────────────────
    step_statuses = {s["step_id"]: s["status"] for s in data["steps"]}
    assert step_statuses[1] == "done", f"Step 1 should be done, got {step_statuses[1]}"
    assert step_statuses[2] == "done", f"Step 2 should be done, got {step_statuses[2]}"

    # ── 5. plan.md content ────────────────────────────────────────────────────
    md_content = open(md_path, encoding="utf-8").read()
    assert "Which music genres" in md_content
    assert "complex_query" in md_content

    print(f"\n[OK] plan files at: {task_dir}")
    print(f"     task_id   = {task_id}")
    print(f"     steps     = {step_statuses}")
    print(f"     plan.md preview:\n{md_content[:400]}")


# ── Test: plan_manager=None → skill still works ────────────────────────────────

def test_complex_skill_works_without_plan_manager(db_and_tools):
    """Skill should run normally when plan_manager is None (graceful degradation)."""
    from langchain_core.messages import HumanMessage
    from skills.complex_query.skill import ComplexQuerySkill

    db_manager, tool_manager = db_and_tools
    llm = _mock_llm_for_complex_query()

    skill = ComplexQuerySkill(
        llm=llm,
        tool_manager=tool_manager,
        db_manager=db_manager,
        retriever=None,
        plan_manager=None,   # explicitly disabled
    )

    result = skill.graph.invoke({
        "messages": [HumanMessage(content="Which genres have the most tracks?")],
        "tables": [],
        "table_schema": "",
        "query_plan": [],
        "step_results": {},
        "plan_completed": False,
        "retrieval_stats": {},
        "task_id": "",
    })

    # Should complete without errors; task_id stays empty
    assert result.get("plan_completed") is True
    assert result.get("task_id", "") == ""


# ── Test: format_for_llm injected into plan node on resume ───────────────────

def test_format_for_llm_injected_on_existing_task(plan_manager):
    """
    If a task_id already exists in state (resume scenario),
    format_for_llm should return the prior progress string.
    """
    tid = plan_manager.new_task_id()
    plan_manager.create_plan(tid, "Resume test", "Long running task", "complex_query", [
        {"step_id": 1, "description": "Fetch IDs", "depends_on": []},
        {"step_id": 2, "description": "Fetch details", "depends_on": [1]},
    ])
    plan_manager.update_step(tid, 1, "done", result_summary="Got 10 IDs")

    ctx = plan_manager.format_for_llm(tid)
    assert "Resume test" in ctx
    assert "Fetch IDs" in ctx
    assert "done" in ctx.lower()
    assert "Fetch details" in ctx   # pending step also shown
