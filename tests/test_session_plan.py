"""
Unit tests for agent/session_plan.py — SessionPlanManager

Tests cover:
  - create_plan / get_plan round-trip
  - update_step (in_progress → done / failed)
  - add_note
  - mark_complete (success & failure)
  - format_for_llm output structure
  - plan.md / plan.json written to disk
  - idempotent get_plan on unknown task_id
"""

import os
import json
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.session_plan import SessionPlanManager, SessionPlan, PlanStep, PlanNote


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pm(tmp_path):
    """A fresh SessionPlanManager pointing at a temp directory."""
    return SessionPlanManager(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def two_step_plan(pm):
    """Create a plan with 2 steps and return (pm, task_id)."""
    tid = pm.new_task_id()
    pm.create_plan(
        task_id=tid,
        title="Test query",
        description="Fetch all employees with salary > 50k",
        skill="complex_query",
        steps=[
            {"step_id": 1, "description": "Get dept IDs", "depends_on": []},
            {"step_id": 2, "description": "Get employees", "depends_on": [1]},
        ],
    )
    return pm, tid


# ── Tests: new_task_id ─────────────────────────────────────────────────────────

def test_new_task_id_unique(pm):
    ids = {pm.new_task_id() for _ in range(20)}
    assert len(ids) == 20, "task IDs should be unique"


def test_new_task_id_length(pm):
    tid = pm.new_task_id()
    assert len(tid) == 12


# ── Tests: create_plan / get_plan ──────────────────────────────────────────────

def test_create_plan_returns_plan(pm):
    tid = pm.new_task_id()
    plan = pm.create_plan(
        task_id=tid,
        title="Title",
        description="Desc",
        skill="complex_query",
        steps=[{"step_id": 1, "description": "Step 1", "depends_on": []}],
    )
    assert isinstance(plan, SessionPlan)
    assert plan.task_id == tid
    assert plan.status == "in_progress"   # create_plan starts as in_progress
    assert len(plan.steps) == 1


def test_get_plan_round_trip(two_step_plan):
    pm, tid = two_step_plan
    plan = pm.get_plan(tid)
    assert plan is not None
    assert plan.task_id == tid
    assert len(plan.steps) == 2
    assert plan.steps[0].status == "pending"
    assert plan.steps[1].status == "pending"


def test_get_plan_unknown_returns_none(pm):
    assert pm.get_plan("nonexistent_id") is None


# ── Tests: update_step ─────────────────────────────────────────────────────────

def test_update_step_in_progress(two_step_plan):
    pm, tid = two_step_plan
    pm.update_step(tid, 1, "in_progress")
    plan = pm.get_plan(tid)
    step = next(s for s in plan.steps if s.step_id == 1)
    assert step.status == "in_progress"
    assert step.started_at is not None


def test_update_step_done_with_metadata(two_step_plan):
    pm, tid = two_step_plan
    pm.update_step(tid, 1, "done", sql="SELECT 1", result_summary="Got 5 rows")
    plan = pm.get_plan(tid)
    step = next(s for s in plan.steps if s.step_id == 1)
    assert step.status == "done"
    assert step.sql == "SELECT 1"
    assert step.result_summary == "Got 5 rows"
    assert step.completed_at is not None


def test_update_step_failed_with_error(two_step_plan):
    pm, tid = two_step_plan
    pm.update_step(tid, 2, "failed", error="Table not found")
    plan = pm.get_plan(tid)
    step = next(s for s in plan.steps if s.step_id == 2)
    assert step.status == "failed"
    assert step.error == "Table not found"


def test_update_step_unknown_step_id_is_noop(two_step_plan):
    """Updating a non-existent step should not raise."""
    pm, tid = two_step_plan
    pm.update_step(tid, 99, "done")  # should not raise
    plan = pm.get_plan(tid)
    assert len(plan.steps) == 2  # original steps unchanged


# ── Tests: add_note ────────────────────────────────────────────────────────────

def test_add_note_basic(two_step_plan):
    pm, tid = two_step_plan
    pm.add_note(tid, "blocker", "DB timeout", "Connection lost after 30s")
    plan = pm.get_plan(tid)
    assert len(plan.notes) == 1
    n = plan.notes[0]
    assert n.note_type == "blocker"
    assert n.title == "DB timeout"
    assert "Connection lost" in n.content


def test_add_multiple_notes(two_step_plan):
    pm, tid = two_step_plan
    pm.add_note(tid, "info", "Note 1", "body 1")
    pm.add_note(tid, "warning", "Note 2", "body 2")
    plan = pm.get_plan(tid)
    assert len(plan.notes) == 2


# ── Tests: mark_complete ───────────────────────────────────────────────────────

def test_mark_complete_success(two_step_plan):
    pm, tid = two_step_plan
    pm.update_step(tid, 1, "done")
    pm.update_step(tid, 2, "done")
    pm.mark_complete(tid, success=True)
    plan = pm.get_plan(tid)
    assert plan.status == "done"


def test_mark_complete_failure(two_step_plan):
    pm, tid = two_step_plan
    pm.update_step(tid, 1, "done")
    pm.update_step(tid, 2, "failed", error="timeout")
    pm.mark_complete(tid, success=False)
    plan = pm.get_plan(tid)
    assert plan.status == "failed"


# ── Tests: format_for_llm ──────────────────────────────────────────────────────

def test_format_for_llm_contains_title(two_step_plan):
    pm, tid = two_step_plan
    ctx = pm.format_for_llm(tid)
    assert "Test query" in ctx


def test_format_for_llm_shows_done_steps(two_step_plan):
    pm, tid = two_step_plan
    pm.update_step(tid, 1, "done", result_summary="5 dept IDs")
    ctx = pm.format_for_llm(tid)
    assert "done" in ctx.lower()
    assert "Get dept IDs" in ctx


def test_format_for_llm_shows_blockers(two_step_plan):
    pm, tid = two_step_plan
    pm.add_note(tid, "blocker", "Index missing", "No index on emp_id")
    ctx = pm.format_for_llm(tid)
    assert "Index missing" in ctx


def test_format_for_llm_unknown_task_returns_empty(pm):
    assert pm.format_for_llm("does_not_exist") == ""


# ── Tests: disk persistence ────────────────────────────────────────────────────

def test_plan_json_written_to_disk(two_step_plan, tmp_path):
    pm, tid = two_step_plan
    json_path = os.path.join(str(tmp_path / "sessions"), tid, "plan.json")
    assert os.path.exists(json_path)
    data = json.loads(open(json_path, encoding="utf-8").read())
    assert data["task_id"] == tid
    assert len(data["steps"]) == 2


def test_plan_md_written_to_disk(two_step_plan, tmp_path):
    pm, tid = two_step_plan
    md_path = pm.get_plan_path(tid)
    assert os.path.exists(md_path)
    content = open(md_path, encoding="utf-8").read()
    assert "Test query" in content
    assert "complex_query" in content


def test_plan_json_persists_across_manager_instances(two_step_plan, tmp_path):
    """A new manager instance reading the same directory should reload state."""
    pm, tid = two_step_plan
    pm.update_step(tid, 1, "done", result_summary="reloaded")

    # New manager instance, same base_dir
    pm2 = SessionPlanManager(base_dir=str(tmp_path / "sessions"))
    plan = pm2.get_plan(tid)
    assert plan is not None
    step = next(s for s in plan.steps if s.step_id == 1)
    assert step.status == "done"
    assert step.result_summary == "reloaded"


# ── Tests: data_analysis 7-step plan ──────────────────────────────────────────

def test_data_analysis_eight_steps(pm):
    tid = pm.new_task_id()
    steps = [
        {"step_id": i, "description": f"Step {i}", "depends_on": [i - 1] if i > 1 else []}
        for i in range(1, 9)
    ]
    plan = pm.create_plan(tid, "Analysis", "Analyze sales", "data_analysis", steps)
    assert len(plan.steps) == 8
    assert plan.skill == "data_analysis"

    # Simulate full run
    for i in range(1, 9):
        pm.update_step(tid, i, "in_progress")
        pm.update_step(tid, i, "done", result_summary=f"step {i} ok")
    pm.mark_complete(tid, success=True)

    final = pm.get_plan(tid)
    assert final.status == "done"
    assert all(s.status == "done" for s in final.steps)
