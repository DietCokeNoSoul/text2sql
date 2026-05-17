"""SQL performance analysis + optimization helpers.

Phase 1:
- EXPLAIN analysis and scoring

Phase 2:
- LLM-based candidate rewrite
- EXPLAIN compare and pick only improved candidate
- Conservative semantic-compatibility guard
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .types import DatabaseDialect

logger = logging.getLogger(__name__)


@dataclass
class SQLPerformanceAnalysis:
    """Structured SQL performance analysis result."""

    enabled: bool
    dialect: str
    explain_sql: str = ""
    raw_plan: str = ""
    uses_index: bool = False
    full_scan: bool = False
    uses_temporary: bool = False
    uses_filesort: bool = False
    estimated_rows: Optional[int] = None
    estimated_cost: Optional[float] = None
    score: int = 0
    issues: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "dialect": self.dialect,
            "explain_sql": self.explain_sql,
            "raw_plan": self.raw_plan,
            "uses_index": self.uses_index,
            "full_scan": self.full_scan,
            "uses_temporary": self.uses_temporary,
            "uses_filesort": self.uses_filesort,
            "estimated_rows": self.estimated_rows,
            "estimated_cost": self.estimated_cost,
            "score": self.score,
            "issues": self.issues,
            "summary": self.summary,
        }


@dataclass
class SQLPerformanceOptimizationResult:
    """Optimization result for a SQL statement."""

    original_sql: str
    final_sql: str
    original_analysis: SQLPerformanceAnalysis
    final_analysis: SQLPerformanceAnalysis
    optimized: bool = False
    attempts: int = 0
    rewrite_candidates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    semantic_check_passed: Optional[bool] = None
    semantic_check_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "original_sql": self.original_sql,
            "final_sql": self.final_sql,
            "original_analysis": self.original_analysis.to_dict(),
            "final_analysis": self.final_analysis.to_dict(),
            "optimized": self.optimized,
            "attempts": self.attempts,
            "rewrite_candidates": self.rewrite_candidates,
            "notes": self.notes,
            "semantic_check_passed": self.semantic_check_passed,
            "semantic_check_reason": self.semantic_check_reason,
        }


class SQLPerformanceAnalyzer:
    """Run EXPLAIN and produce heuristic index/performance diagnostics."""

    def __init__(
        self,
        db_manager,
        llm=None,
        enabled: bool = True,
        optimize_enabled: bool = False,
        rows_warning_threshold: int = 10000,
        min_score_improvement: int = 8,
        max_rewrite_rounds: int = 1,
        optimize_score_threshold: int = 70,
        semantic_validation_enabled: bool = True,
        semantic_sample_rows: int = 20,
        optimize_trigger_low_score: bool = True,
        optimize_trigger_large_rows: bool = True,
        optimize_trigger_full_scan: bool = True,
        optimize_trigger_filesort: bool = True,
        optimize_trigger_temporary: bool = True,
        optimize_trigger_high_cost: bool = False,
        optimize_min_triggers: int = 1,
        cost_warning_threshold: float = 1000.0,
    ):
        self.db_manager = db_manager
        self.llm = llm
        self.enabled = enabled
        self.optimize_enabled = optimize_enabled
        self.rows_warning_threshold = rows_warning_threshold
        self.min_score_improvement = min_score_improvement
        self.max_rewrite_rounds = max_rewrite_rounds
        self.optimize_score_threshold = optimize_score_threshold
        self.semantic_validation_enabled = semantic_validation_enabled
        self.semantic_sample_rows = semantic_sample_rows
        self.optimize_trigger_low_score = optimize_trigger_low_score
        self.optimize_trigger_large_rows = optimize_trigger_large_rows
        self.optimize_trigger_full_scan = optimize_trigger_full_scan
        self.optimize_trigger_filesort = optimize_trigger_filesort
        self.optimize_trigger_temporary = optimize_trigger_temporary
        self.optimize_trigger_high_cost = optimize_trigger_high_cost
        self.optimize_min_triggers = max(1, int(optimize_min_triggers))
        self.cost_warning_threshold = max(0.0, float(cost_warning_threshold))

    def analyze(self, sql: str) -> SQLPerformanceAnalysis:
        """Analyze SQL plan with EXPLAIN. Never raises to caller."""
        dialect = self.db_manager.get_dialect()
        analysis = SQLPerformanceAnalysis(
            enabled=self.enabled,
            dialect=dialect.value,
            score=0,
        )

        if not self.enabled:
            analysis.summary = "SQL性能分析已关闭"
            return analysis

        if not sql or not sql.strip():
            analysis.summary = "SQL为空，跳过性能分析"
            return analysis

        try:
            explain_sql = self._build_explain_sql(sql, dialect)
            analysis.explain_sql = explain_sql
            raw_plan = self.db_manager.db.run(explain_sql)
            analysis.raw_plan = str(raw_plan)

            self._parse_plan_text(analysis)
            self._score(analysis)
            analysis.summary = self._build_summary(analysis)
            return analysis
        except Exception as e:
            logger.warning("[SQLPerf] EXPLAIN failed: %s", e)
            analysis.summary = f"EXPLAIN分析失败: {e}"
            return analysis

    def optimize_sql(self, sql: str, constraints: Optional[List[str]] = None) -> SQLPerformanceOptimizationResult:
        """Optimize SQL via explain-guided LLM rewrite (conservative)."""
        baseline = self.analyze(sql)
        result = SQLPerformanceOptimizationResult(
            original_sql=sql,
            final_sql=sql,
            original_analysis=baseline,
            final_analysis=baseline,
            optimized=False,
            attempts=0,
        )

        if not self.enabled:
            result.notes.append("性能分析未启用，跳过优化")
            return result
        if not self.optimize_enabled:
            result.notes.append("自动优化未启用，跳过优化")
            return result
        if self.llm is None:
            result.notes.append("未提供LLM，跳过自动优化")
            return result
        if not self._should_optimize(baseline):
            result.notes.append("当前计划无明显瓶颈，跳过重写")
            return result

        best_sql = sql
        best_analysis = baseline
        current_sql = sql

        for _ in range(max(1, self.max_rewrite_rounds)):
            result.attempts += 1
            candidate = self._rewrite_sql_with_llm(
                current_sql=current_sql,
                analysis=best_analysis,
                constraints=constraints or [],
            )
            if not candidate:
                result.notes.append("候选SQL为空，停止优化")
                break
            candidate = candidate.strip().rstrip(";")
            result.rewrite_candidates.append(candidate)

            if not self._looks_like_select(candidate):
                result.notes.append("候选SQL非SELECT，拒绝")
                continue
            if not self._semantically_compatible(best_sql, candidate):
                result.notes.append("候选SQL语义守卫未通过，拒绝")
                continue

            candidate_analysis = self.analyze(candidate)
            gain = candidate_analysis.score - best_analysis.score
            if gain >= self.min_score_improvement:
                if self.semantic_validation_enabled:
                    passed, reason = self._validate_semantic_equivalence(best_sql, candidate)
                    result.semantic_check_passed = passed
                    result.semantic_check_reason = reason
                    if not passed:
                        result.notes.append(f"语义校验未通过: {reason}")
                        continue

                best_sql = candidate
                best_analysis = candidate_analysis
                current_sql = candidate
                result.notes.append(f"接受候选SQL，评分提升 {gain}")
            else:
                result.notes.append(f"候选SQL提升不足({gain})，拒绝")

        result.final_sql = best_sql
        result.final_analysis = best_analysis
        result.optimized = best_sql.strip() != sql.strip()
        return result

    @staticmethod
    def _build_explain_sql(sql: str, dialect: DatabaseDialect) -> str:
        """Build EXPLAIN SQL based on database dialect."""
        normalized = sql.strip().rstrip(";")

        if dialect == DatabaseDialect.SQLITE:
            return f"EXPLAIN QUERY PLAN {normalized}"
        elif dialect == DatabaseDialect.MYSQL:
            return f"EXPLAIN FORMAT=JSON {normalized}"
        elif dialect == DatabaseDialect.POSTGRESQL:
            return f"EXPLAIN (ANALYZE, BUFFERS) {normalized}"
        elif dialect == DatabaseDialect.MSSQL:
            return f"SET SHOWPLAN_XML ON; {normalized}; SET SHOWPLAN_XML OFF;"
        elif dialect == DatabaseDialect.ORACLE:
            return f"EXPLAIN PLAN FOR {normalized}"
        else:
            return f"EXPLAIN {normalized}"

    def _parse_plan_text(self, analysis: SQLPerformanceAnalysis) -> None:
        text = analysis.raw_plan or ""
        up = text.upper()

        analysis.uses_index = bool(
            re.search(r"USING\s+(COVERING\s+)?INDEX", up)
            or re.search(r"SEARCH\s+\w+\s+USING", up)
            or re.search(r"KEY\s*=\s*['\"]?\w+", up)
        )
        analysis.full_scan = bool(
            re.search(r"\bTABLE\s+SCAN\b", up)
            or re.search(r"\bFULL\s+SCAN\b", up)
            or re.search(r"\bTYPE\s*[:=]\s*['\"]?ALL\b", up)
            or ("SCAN " in up and "USING INDEX" not in up and "SEARCH " not in up)
        )
        analysis.uses_temporary = bool(
            "USING TEMPORARY" in up
            or "USE TEMP B-TREE" in up
            or "TEMP TABLE" in up
        )
        analysis.uses_filesort = bool(
            "USING FILESORT" in up
            or "USE TEMP B-TREE FOR ORDER BY" in up
            or "FILESORT" in up
        )

        rows_matchers = [
            re.search(r"\bROWS\s*[:=]\s*(\d+)", up),
            re.search(r"'ROWS'\s*:\s*(\d+)", up),
            re.search(r'"ROWS"\s*:\s*(\d+)', up),
        ]
        for m in rows_matchers:
            if m:
                try:
                    analysis.estimated_rows = int(m.group(1))
                    break
                except Exception:
                    pass

        cost_matchers = [
            re.search(r"['\"]QUERY_COST['\"]\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", up),
            re.search(r"\bCOST\b\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", up),
        ]
        for m in cost_matchers:
            if m:
                try:
                    analysis.estimated_cost = float(m.group(1))
                    break
                except Exception:
                    pass

        if analysis.full_scan:
            analysis.issues.append("检测到全表扫描")
        if not analysis.uses_index:
            analysis.issues.append("未检测到索引命中")
        if analysis.uses_temporary:
            analysis.issues.append("检测到临时表")
        if analysis.uses_filesort:
            analysis.issues.append("检测到额外排序(filesort)")
        if analysis.estimated_rows and analysis.estimated_rows > self.rows_warning_threshold:
            analysis.issues.append(
                f"预计扫描行数较高({analysis.estimated_rows})"
            )
        if analysis.estimated_cost and analysis.estimated_cost > self.cost_warning_threshold:
            analysis.issues.append(f"预计代价较高({analysis.estimated_cost:.2f})")

    def _should_optimize(self, analysis: SQLPerformanceAnalysis) -> bool:
        """Configurable trigger strategy for deciding whether to optimize."""
        triggers: List[str] = []

        if self.optimize_trigger_low_score and analysis.score < self.optimize_score_threshold:
            triggers.append("low_score")
        if self.optimize_trigger_full_scan and analysis.full_scan and not analysis.uses_index:
            triggers.append("full_scan")
        if self.optimize_trigger_temporary and analysis.uses_temporary:
            triggers.append("temporary")
        if self.optimize_trigger_filesort and analysis.uses_filesort:
            triggers.append("filesort")
        if (
            self.optimize_trigger_large_rows
            and analysis.estimated_rows is not None
            and analysis.estimated_rows > self.rows_warning_threshold
        ):
            triggers.append("large_rows")
        if (
            self.optimize_trigger_high_cost
            and analysis.estimated_cost is not None
            and analysis.estimated_cost > self.cost_warning_threshold
        ):
            triggers.append("high_cost")

        should_optimize = len(triggers) >= self.optimize_min_triggers
        logger.info(
            "[SQLPerf] optimize triggers=%s, min=%s, decision=%s",
            triggers,
            self.optimize_min_triggers,
            should_optimize,
        )
        return should_optimize

    def _rewrite_sql_with_llm(
        self,
        current_sql: str,
        analysis: SQLPerformanceAnalysis,
        constraints: List[str],
    ) -> str:
        """Ask LLM to generate a faster, semantically equivalent SQL."""
        dialect = self.db_manager.get_dialect().value
        constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "（无）"
        issues_text = "；".join(analysis.issues) if analysis.issues else "（未识别明显问题）"

        system_prompt = f"""你是一个 SQL 性能优化专家。
目标：在保持语义等价前提下优化 SQL 性能。

硬约束：
1) 只允许输出一条 SELECT SQL，不要 Markdown，不要解释。
2) 不改变业务语义（统计口径/过滤条件/分组维度/排序意图保持一致）。
3) 尽量利用索引：优先避免函数包裹索引列、减少不必要子查询、避免 SELECT *。
4) 数据库方言：{dialect}
5) 用户禁令：
{constraints_text}
"""

        user_prompt = f"""当前 SQL：
{current_sql}

当前执行计划问题：
{issues_text}

请返回优化后的单条 SQL。"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            text = str(getattr(response, "content", "") or "").strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if len(lines) >= 3:
                    text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("sql\n"):
                text = text[4:].strip()
            return text
        except Exception as e:
            logger.warning("[SQLPerf] LLM rewrite failed: %s", e)
            return ""

    @staticmethod
    def _looks_like_select(sql: str) -> bool:
        return bool(re.match(r"^\s*SELECT\b", sql, flags=re.IGNORECASE))

    @staticmethod
    def _normalize_space(sql: str) -> str:
        return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()

    def _extract_table_set(self, sql: str) -> set:
        """Extract table names after FROM/JOIN as a conservative semantic signal."""
        names = set()
        for m in re.finditer(r"\b(?:from|join)\s+([`\"\[]?[\w\.]+[`\"\]]?)", sql, flags=re.IGNORECASE):
            raw = m.group(1).strip("`\"[]")
            if raw:
                names.add(raw.lower())
        return names

    def _semantically_compatible(self, original_sql: str, candidate_sql: str) -> bool:
        """Conservative semantic guard to avoid aggressive rewrites."""
        o = self._normalize_space(original_sql)
        c = self._normalize_space(candidate_sql)

        if o == c:
            return False

        # DISTINCT intent should be preserved.
        if (" select distinct " in f" {o} ") != (" select distinct " in f" {c} "):
            return False

        # Preserve major clauses presence.
        for token in (" group by ", " having "):
            if (token in f" {o} ") != (token in f" {c} "):
                return False

        # Candidate should keep same table set (or superset only for derived optimization avoidance).
        orig_tables = self._extract_table_set(original_sql)
        cand_tables = self._extract_table_set(candidate_sql)
        if orig_tables and cand_tables and orig_tables != cand_tables:
            return False

        return True

    def _validate_semantic_equivalence(self, original_sql: str, candidate_sql: str) -> tuple[bool, str]:
        """Enhanced semantic equivalence validation for different SQL types."""
        try:
            dialect = self.db_manager.get_dialect()
            if dialect not in (DatabaseDialect.SQLITE, DatabaseDialect.MYSQL, DatabaseDialect.POSTGRESQL):
                return True, "当前方言跳过样本语义校验"

            sample_n = max(1, int(self.semantic_sample_rows))

            # Detect SQL type
            is_aggregate = re.search(r"\b(SUM|AVG|COUNT|MIN|MAX)\b", original_sql, re.IGNORECASE)
            has_order_by = " order by " in original_sql.lower()

            if is_aggregate:
                # For aggregate queries, compare aggregate results
                wrapped_original = f"SELECT * FROM ({original_sql.strip().rstrip(';')}) AS _perf_check"
                wrapped_candidate = f"SELECT * FROM ({candidate_sql.strip().rstrip(';')}) AS _perf_check"
            elif has_order_by:
                # For ORDER BY queries, compare Top-K results
                wrapped_original = f"SELECT * FROM ({original_sql.strip().rstrip(';')}) AS _perf_check LIMIT {sample_n}"
                wrapped_candidate = f"SELECT * FROM ({candidate_sql.strip().rstrip(';')}) AS _perf_check LIMIT {sample_n}"
            else:
                # Default: Compare sample rows
                wrapped_original = self._wrap_sample_query(original_sql, sample_n)
                wrapped_candidate = self._wrap_sample_query(candidate_sql, sample_n)

            raw_a = str(self.db_manager.db.run(wrapped_original))
            raw_b = str(self.db_manager.db.run(wrapped_candidate))

            fa = self._result_fingerprint(raw_a)
            fb = self._result_fingerprint(raw_b)
            if fa == fb:
                return True, f"样本结果一致（前{sample_n}行）"
            return False, f"样本结果不一致（前{sample_n}行）"
        except Exception as e:
            return False, f"语义样本校验异常: {e}"

    @staticmethod
    def _wrap_sample_query(sql: str, sample_n: int) -> str:
        base = sql.strip().rstrip(";")
        return f"SELECT * FROM ({base}) AS _perf_semantic_check LIMIT {sample_n}"

    @staticmethod
    def _result_fingerprint(raw: str) -> str:
        # Normalize whitespace and decimal noise conservatively for textual compare.
        text = re.sub(r"\s+", " ", str(raw).strip())
        text = re.sub(r"\.0+([,\]\)])", r"\1", text)
        return text

    def _score(self, analysis: SQLPerformanceAnalysis) -> None:
        score = 100
        if analysis.full_scan:
            score -= 45
        if not analysis.uses_index:
            score -= 20
        if analysis.uses_temporary:
            score -= 15
        if analysis.uses_filesort:
            score -= 15

        if analysis.uses_index and not analysis.full_scan:
            score += 8

        if analysis.estimated_rows and analysis.estimated_rows > self.rows_warning_threshold:
            overflow = analysis.estimated_rows - self.rows_warning_threshold
            penalty = min(20, max(5, overflow // max(1, self.rows_warning_threshold // 4)))
            score -= int(penalty)

        if analysis.estimated_cost and analysis.estimated_cost > self.cost_warning_threshold:
            cost_overflow = analysis.estimated_cost - self.cost_warning_threshold
            ratio = cost_overflow / max(1.0, self.cost_warning_threshold)
            cost_penalty = min(15, max(3, int(ratio * 10)))
            score -= int(cost_penalty)

        analysis.score = max(0, min(100, int(score)))

    @staticmethod
    def _build_summary(analysis: SQLPerformanceAnalysis) -> str:
        idx = "命中索引" if analysis.uses_index else "未命中索引"
        scan = "存在全表扫描" if analysis.full_scan else "无明显全表扫描"
        rows = (
            f"预计扫描行数={analysis.estimated_rows}"
            if analysis.estimated_rows is not None
            else "预计扫描行数未知"
        )
        cost = (
            f"，预计代价={analysis.estimated_cost:.2f}"
            if analysis.estimated_cost is not None
            else ""
        )
        extra = []
        if analysis.uses_temporary:
            extra.append("临时表")
        if analysis.uses_filesort:
            extra.append("filesort")
        extra_text = f"，额外操作={','.join(extra)}" if extra else ""
        return f"性能评分={analysis.score}/100，{idx}，{scan}，{rows}{cost}{extra_text}"
