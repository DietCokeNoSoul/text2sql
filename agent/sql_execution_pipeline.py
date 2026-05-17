"""Unified SQL execution pipeline.

Pipeline order:
1) Static checks are handled by caller before entering this module.
2) Explain analysis + optional optimization.
3) Decide direct execute or correction path on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from agent.sql_correction import execute_with_correction
from agent.sql_performance import SQLPerformanceAnalyzer, SQLPerformanceOptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class SQLExecutionPipelineResult:
    """Result returned by unified execution pipeline."""

    success: bool
    final_sql: str
    result: str
    error: str
    retries: int
    decision: str
    used_correction: bool
    correction_trace: List[Dict[str, str]]
    perf_optimization: SQLPerformanceOptimizationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_sql": self.final_sql,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
            "decision": self.decision,
            "used_correction": self.used_correction,
            "correction_trace": self.correction_trace,
            "perf_optimization": self.perf_optimization.to_dict(),
        }


def _build_lineage(opt_result: SQLPerformanceOptimizationResult) -> List[str]:
    return [
        opt_result.original_sql,
        *opt_result.rewrite_candidates,
        opt_result.final_sql,
    ]


def run_sql_execution_pipeline(
    *,
    sql: str,
    query_tool: Any,
    db_manager: Any,
    llm: BaseChatModel,
    perf_analyzer: SQLPerformanceAnalyzer,
    constraints: Optional[List[str]] = None,
    context_label: str = "",
    correction_max_retries: int = 2,
    precomputed_optimization: Optional[SQLPerformanceOptimizationResult] = None,
) -> SQLExecutionPipelineResult:
    """Run unified SQL execution pipeline.

    Decision contract:
    - ``direct_execute``: explain indicates no rewrite adopted, execute directly.
    - ``optimize_then_execute``: explain/LLM optimization adopted, execute optimized SQL.
    - ``correct_after_failure``: initial execution failed, retry with category-aware correction.
    """
    label = context_label or "[SQLPipeline]"
    constraints = constraints or []

    # Step 2: Explain analysis + optional optimization.
    perf_opt_result = precomputed_optimization or perf_analyzer.optimize_sql(sql, constraints=constraints)
    sql_to_execute = perf_opt_result.final_sql
    decision = "optimize_then_execute" if perf_opt_result.optimized else "direct_execute"

    logger.info(
        "%s decision=%s score=%s->%s",
        label,
        decision,
        perf_opt_result.original_analysis.score,
        perf_opt_result.final_analysis.score,
    )

    # Step 3a: direct execution first.
    try:
        result = query_tool.invoke({"query": sql_to_execute})
        return SQLExecutionPipelineResult(
            success=True,
            final_sql=sql_to_execute,
            result=result,
            error="",
            retries=0,
            decision=decision,
            used_correction=False,
            correction_trace=[],
            perf_optimization=perf_opt_result,
        )
    except Exception as first_error:
        logger.warning("%s first execution failed: %s", label, str(first_error)[:200])

    # Step 3b: correction fallback.
    correction_result = execute_with_correction(
        sql=sql_to_execute,
        query_tool=query_tool,
        db_manager=db_manager,
        llm=llm,
        max_retries=correction_max_retries,
        context_label=label,
        rewrite_lineage=_build_lineage(perf_opt_result),
        prefer_minimal_fix=perf_opt_result.optimized,
    )

    return SQLExecutionPipelineResult(
        success=bool(correction_result.get("success", False)),
        final_sql=str(correction_result.get("final_sql", sql_to_execute)),
        result=str(correction_result.get("result", "")),
        error=str(correction_result.get("error", "")),
        retries=int(correction_result.get("retries", correction_max_retries)),
        decision="correct_after_failure",
        used_correction=True,
        correction_trace=list(correction_result.get("correction_trace", [])),
        perf_optimization=perf_opt_result,
    )
