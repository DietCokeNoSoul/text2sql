"""Session Plan Manager — 会话任务计划追踪器。

为每个 ComplexQuery / DataAnalysis 会话创建独立的结构化文件，记录：
- 任务分解与步骤状态（pending / in_progress / done / failed）
- 每步执行的 SQL、结果摘要、错误信息
- 阻塞点、关键注记（task_state / blocker / info / warning / result）
- 整体任务完成进度

存储路径: {base_dir}/{task_id}/plan.md  +  plan.json（机器可读）

plan 节点可调用 format_for_llm() 将当前进度注入 LLM 上下文，
实现任务链追踪与跨轮次会话恢复。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

StepStatus = Literal["pending", "in_progress", "done", "failed", "skipped"]
NoteType   = Literal["task_state", "blocker", "info", "warning", "result"]

_STATUS_ICON: Dict[str, str] = {
    "pending":     "[pending]",
    "in_progress": "[running]",
    "done":        "[done]",
    "failed":      "[FAILED]",
    "skipped":     "[skipped]",
}
_NOTE_ICON: Dict[str, str] = {
    "blocker":    "[BLOCKER]",
    "task_state": "[task]",
    "info":       "[info]",
    "warning":    "[warn]",
    "result":     "[result]",
}


# ── Data models ────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    step_id: int
    description: str
    status: StepStatus = "pending"
    sql: str = ""
    result_summary: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    notes: str = ""


@dataclass
class PlanNote:
    note_type: NoteType
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class SessionPlan:
    task_id: str
    title: str
    description: str
    skill: str
    status: Literal["pending", "in_progress", "done", "failed"] = "in_progress"
    created_at: str = ""
    updated_at: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    notes: List[PlanNote] = field(default_factory=list)


# ── Manager ────────────────────────────────────────────────────────────────

class SessionPlanManager:
    """每会话任务计划管理器。

    用法示例：
        mgr = SessionPlanManager("report/sessions")
        task_id = mgr.new_task_id()

        mgr.create_plan(task_id, "分析艺术家销售", "...", "complex_query",
                        steps=[{"step_id": 1, "description": "获取艺术家列表", "query": "SELECT..."},
                               {"step_id": 2, "description": "统计专辑数量",   "query": "SELECT..."}])

        mgr.update_step(task_id, 1, "in_progress")
        mgr.update_step(task_id, 1, "done", result_summary="返回 47 条记录")
        mgr.add_note(task_id, "task_state", "Step 1 完成", "艺术家列表已获取")
        mgr.mark_complete(task_id)

        # 将进度注入 LLM 上下文
        context = mgr.format_for_llm(task_id)
    """

    def __init__(self, base_dir: str = "report/sessions") -> None:
        self._base_dir = Path(base_dir)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def new_task_id(self) -> str:
        """生成唯一短 ID（12 位十六进制）。"""
        return uuid.uuid4().hex[:12]

    def create_plan(
        self,
        task_id: str,
        title: str,
        description: str,
        skill: str,
        steps: Optional[List[dict]] = None,
    ) -> SessionPlan:
        """创建并持久化新的任务计划。

        Args:
            task_id:     唯一任务 ID
            title:       简短任务标题
            description: 用户原始问题或任务说明
            skill:       "complex_query" | "data_analysis"
            steps:       步骤列表，每项含 step_id / description / query（可选）
        """
        now = _now()
        plan = SessionPlan(
            task_id=task_id,
            title=title,
            description=description,
            skill=skill,
            status="in_progress",
            created_at=now,
            updated_at=now,
            steps=[
                PlanStep(
                    step_id=s.get("step_id", i + 1),
                    description=s.get("description", ""),
                    sql=s.get("query", ""),
                )
                for i, s in enumerate(steps or [])
            ],
        )
        self._save(plan)
        logger.info(f"[SessionPlan] Created plan {task_id}: '{title}' ({len(plan.steps)} steps)")
        return plan

    def get_plan(self, task_id: str) -> Optional[SessionPlan]:
        """从磁盘读取已有任务计划；不存在则返回 None。"""
        json_file = self._base_dir / task_id / "plan.json"
        if not json_file.exists():
            return None
        try:
            return _from_dict(json.loads(json_file.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"[SessionPlan] Failed to read plan {task_id}: {e}")
            return None

    def update_step(
        self,
        task_id: str,
        step_id: int,
        status: StepStatus,
        sql: str = "",
        result_summary: str = "",
        error: str = "",
        notes: str = "",
    ) -> None:
        """更新某步骤的执行状态和结果摘要。"""
        plan = self.get_plan(task_id)
        if plan is None:
            return
        now = _now()
        for step in plan.steps:
            if step.step_id == step_id:
                step.status = status
                if sql:
                    step.sql = sql
                if result_summary:
                    step.result_summary = result_summary[:300]
                if error:
                    step.error = error[:200]
                if notes:
                    step.notes = notes
                if status == "in_progress" and not step.started_at:
                    step.started_at = now
                if status in ("done", "failed", "skipped"):
                    step.completed_at = now
                break
        plan.updated_at = now
        self._save(plan)

    def add_note(
        self,
        task_id: str,
        note_type: NoteType,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> None:
        """添加任务注记（阻塞点、里程碑、洞察摘要等）。"""
        plan = self.get_plan(task_id)
        if plan is None:
            return
        plan.notes.append(PlanNote(
            note_type=note_type,
            title=title,
            content=content[:500],
            tags=tags or [],
            created_at=_now(),
        ))
        plan.updated_at = _now()
        self._save(plan)

    def mark_complete(self, task_id: str, success: bool = True) -> None:
        """将整个任务标记为完成或失败。"""
        plan = self.get_plan(task_id)
        if plan is None:
            return
        plan.status = "done" if success else "failed"
        plan.updated_at = _now()
        self._save(plan)
        logger.info(f"[SessionPlan] Plan {task_id} marked {'done' if success else 'failed'}")

    # ── LLM context ────────────────────────────────────────────────────────

    def format_for_llm(self, task_id: str) -> str:
        """将当前计划进度格式化为 LLM 可理解的上下文摘要。

        plan 节点在生成/续写计划前调用此方法，将结果拼接到 system prompt，
        使 LLM 知晓哪些步骤已完成、哪些待执行。
        """
        plan = self.get_plan(task_id)
        if plan is None:
            return ""

        done       = [s for s in plan.steps if s.status == "done"]
        in_progress = [s for s in plan.steps if s.status == "in_progress"]
        pending    = [s for s in plan.steps if s.status == "pending"]
        failed     = [s for s in plan.steps if s.status == "failed"]

        lines = [
            f"## 当前任务进度: {plan.title}",
            f"进度: {len(done)}/{len(plan.steps)} 步完成 | 任务状态: {plan.status}",
            "",
        ]

        if done:
            lines.append("### [done] 已完成步骤")
            for s in done:
                lines.append(f"- Step {s.step_id}: {s.description}")
                if s.result_summary:
                    lines.append(f"  → 结果: {s.result_summary}")

        if in_progress:
            lines.append("### [running] 执行中步骤")
            for s in in_progress:
                lines.append(f"- Step {s.step_id}: {s.description}")
                if s.started_at:
                    lines.append(f"  → 开始时间: {s.started_at}")

        if pending:
            lines.append("### [pending] 待执行步骤")
            for s in pending:
                lines.append(f"- Step {s.step_id}: {s.description}")

        if failed:
            lines.append("### [FAILED] 失败步骤")
            for s in failed:
                lines.append(f"- Step {s.step_id}: {s.description} — {s.error}")

        blockers = [n for n in plan.notes if n.note_type == "blocker"]
        if blockers:
            lines.append("### [BLOCKER] 阻塞点")
            for n in blockers:
                lines.append(f"- [{n.created_at}] {n.title}: {n.content}")

        return "\n".join(lines)

    def get_plan_path(self, task_id: str) -> str:
        """返回 plan.md 文件路径字符串。"""
        return str(self._base_dir / task_id / "plan.md")

    # ── Internal ──────────────────────────────────────────────────────────

    def _save(self, plan: SessionPlan) -> None:
        plan_dir = self._base_dir / plan.task_id
        plan_dir.mkdir(parents=True, exist_ok=True)

        # Machine-readable JSON
        (plan_dir / "plan.json").write_text(
            json.dumps(_to_dict(plan), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Human-readable Markdown
        (plan_dir / "plan.md").write_text(
            _to_markdown(plan), encoding="utf-8"
        )


# ── Pure functions ─────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_dict(plan: SessionPlan) -> dict:
    return {
        "task_id":    plan.task_id,
        "title":      plan.title,
        "description": plan.description,
        "skill":      plan.skill,
        "status":     plan.status,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "steps": [
            {
                "step_id":        s.step_id,
                "description":    s.description,
                "status":         s.status,
                "sql":            s.sql,
                "result_summary": s.result_summary,
                "error":          s.error,
                "started_at":     s.started_at,
                "completed_at":   s.completed_at,
                "notes":          s.notes,
            }
            for s in plan.steps
        ],
        "notes": [
            {
                "note_type":  n.note_type,
                "title":      n.title,
                "content":    n.content,
                "tags":       n.tags,
                "created_at": n.created_at,
            }
            for n in plan.notes
        ],
    }


def _from_dict(data: dict) -> SessionPlan:
    return SessionPlan(
        task_id=data["task_id"],
        title=data["title"],
        description=data.get("description", ""),
        skill=data["skill"],
        status=data["status"],
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        steps=[
            PlanStep(
                step_id=s["step_id"],
                description=s["description"],
                status=s.get("status", "pending"),
                sql=s.get("sql", ""),
                result_summary=s.get("result_summary", ""),
                error=s.get("error", ""),
                started_at=s.get("started_at", ""),
                completed_at=s.get("completed_at", ""),
                notes=s.get("notes", ""),
            )
            for s in data.get("steps", [])
        ],
        notes=[
            PlanNote(
                note_type=n["note_type"],
                title=n["title"],
                content=n.get("content", ""),
                tags=n.get("tags", []),
                created_at=n.get("created_at", ""),
            )
            for n in data.get("notes", [])
        ],
    )


def _to_markdown(plan: SessionPlan) -> str:
    lines = [
        f"# {plan.title}",
        "",
        f"**Task ID**: `{plan.task_id}`  ",
        f"**Skill**: `{plan.skill}`  ",
        f"**Status**: {plan.status}  ",
        f"**Created**: {plan.created_at}  ",
        f"**Updated**: {plan.updated_at}",
        "",
        "## 任务描述",
        "",
        plan.description,
        "",
        "## 步骤进度",
        "",
        "| 步骤 | 描述 | 状态 | 开始时间 | 完成时间 |",
        "|------|------|------|----------|----------|",
    ]

    for s in plan.steps:
        icon      = _STATUS_ICON.get(s.status, "?")
        started   = s.started_at or "-"
        completed = s.completed_at or "-"
        lines.append(f"| Step {s.step_id} | {s.description} | {icon} {s.status} | {started} | {completed} |")

    lines.append("")

    # Execution detail sections
    for s in plan.steps:
        if s.status in ("done", "failed", "in_progress"):
            icon = _STATUS_ICON.get(s.status, "?")
            lines += [
                f"### Step {s.step_id} — {s.description} {icon}",
                "",
            ]
            if s.sql:
                lines += ["**SQL**:", "```sql", s.sql, "```", ""]
            if s.result_summary:
                lines += [f"**结果**: {s.result_summary}", ""]
            if s.error:
                lines += [f"**错误**: `{s.error}`", ""]
            if s.notes:
                lines += [f"**备注**: {s.notes}", ""]

    # Notes section
    if plan.notes:
        lines += ["## 任务注记", ""]
        for n in plan.notes:
            icon = _NOTE_ICON.get(n.note_type, "[note]")
            tag_str = " ".join(f"`{t}`" for t in n.tags) if n.tags else ""
            lines += [
                f"### {icon} {n.title}",
                f"**类型**: {n.note_type}  **时间**: {n.created_at}  {tag_str}",
                "",
                n.content,
                "",
            ]

    return "\n".join(lines)
