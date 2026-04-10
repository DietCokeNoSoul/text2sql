"""Matplotlib PNG 图表生成器。

支持图表类型:
  - bar:   竖向柱状图
  - pie:   饼图
  - line:  折线图

输出格式: PNG 文件 (可在浏览器/Markdown 预览中直接显示)

用法::

    from skills.data_analysis.chart_generator import ChartGenerator, ChartSpec

    spec = ChartSpec(
        chart_type="bar",
        title="商店数量按类型分布",
        labels=["Type 1", "Type 2"],
        values=[9.0, 5.0],
        x_label="商店类型",
        y_label="数量",
    )
    path = ChartGenerator.render(spec, output_dir="/tmp/charts")
"""

from __future__ import annotations

import ast
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

# ── CJK font detection ────────────────────────────────────────────────────────

_CJK_FONT: Optional[str] = None
for _name in ["SimHei", "Microsoft YaHei", "PingFang SC",
              "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
    if any(_name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _CJK_FONT = _name
        break


def _setup_style() -> None:
    """Apply consistent style and CJK font (if available)."""
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    if _CJK_FONT:
        plt.rcParams["font.family"] = _CJK_FONT


def _to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ── Data spec ────────────────────────────────────────────────────────────────

@dataclass
class ChartSpec:
    """图表规格。"""
    chart_type: str          # "bar" | "pie" | "line"
    title: str
    labels: List[str]
    values: List[float]
    x_label: str = ""
    y_label: str = ""
    width: int = 700
    height: int = 420


# ── Renderers (return PNG bytes) ──────────────────────────────────────────────

def _bar_chart(spec: ChartSpec) -> bytes:
    _setup_style()
    w, h = spec.width / 100, spec.height / 100
    fig, ax = plt.subplots(figsize=(w, h))

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(spec.labels))]
    bars = ax.bar(spec.labels, spec.values, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels on top
    for bar, val in zip(bars, spec.values):
        label = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(spec.values) * 0.01,
                label, ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_title(spec.title, fontsize=13, fontweight="bold", pad=10)
    if spec.x_label:
        ax.set_xlabel(spec.x_label, fontsize=10)
    if spec.y_label:
        ax.set_ylabel(spec.y_label, fontsize=10)

    # Rotate long labels
    if spec.labels and max(len(str(l)) for l in spec.labels) > 6:
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    data = _to_png_bytes(fig)
    plt.close(fig)
    return data


def _pie_chart(spec: ChartSpec) -> bytes:
    _setup_style()
    w, h = spec.width / 100, spec.height / 100
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_facecolor("white")

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(spec.labels))]
    wedges, texts, autotexts = ax.pie(
        spec.values,
        labels=spec.labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("white")
        t.set_fontweight("bold")

    ax.set_title(spec.title, fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    data = _to_png_bytes(fig)
    plt.close(fig)
    return data


def _line_chart(spec: ChartSpec) -> bytes:
    _setup_style()
    w, h = spec.width / 100, spec.height / 100
    fig, ax = plt.subplots(figsize=(w, h))

    color = _PALETTE[0]
    x_idx = list(range(len(spec.labels)))
    ax.plot(x_idx, spec.values, marker="o", color=color, linewidth=2.5,
            markersize=7, markerfacecolor="white", markeredgewidth=2,
            markeredgecolor=color, zorder=3)
    ax.fill_between(x_idx, spec.values, alpha=0.12, color=color)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(spec.labels,
                       rotation=20 if max(len(str(l)) for l in spec.labels) > 6 else 0)
    ax.set_title(spec.title, fontsize=13, fontweight="bold", pad=10)
    if spec.x_label:
        ax.set_xlabel(spec.x_label, fontsize=10)
    if spec.y_label:
        ax.set_ylabel(spec.y_label, fontsize=10)

    fig.tight_layout()
    data = _to_png_bytes(fig)
    plt.close(fig)
    return data




def _parse_query_result(raw: str) -> Tuple[List[str], List[float]]:
    """
    Parse a SQL query result string into (labels, values).

    Handles formats returned by LangChain SQLDatabase:
      [(label1, val1), (label2, val2), ...]
      [(val,), ...]
    """
    raw = raw.strip()
    # Normalize DB-specific types that ast.literal_eval cannot handle
    raw = re.sub(r"Decimal\('([\d.]+)'\)", r"\1", raw)
    raw = re.sub(r"datetime\.date\([^)]+\)", "'date'", raw)
    try:
        rows = ast.literal_eval(raw)
    except Exception:
        return [], []

    if not isinstance(rows, list) or not rows:
        return [], []

    labels: List[str] = []
    values: List[float] = []

    for row in rows:
        if isinstance(row, (int, float)):
            labels.append(str(len(labels) + 1))
            values.append(float(row))
        elif isinstance(row, (list, tuple)):
            if len(row) == 1:
                try:
                    labels.append(str(len(labels) + 1))
                    values.append(float(row[0]))
                except (TypeError, ValueError):
                    pass
            elif len(row) >= 2:
                try:
                    labels.append(str(row[0]))
                    values.append(float(row[-1]))
                except (TypeError, ValueError):
                    pass

    return labels, values


# ── Public API ────────────────────────────────────────────────────────────────

class ChartGenerator:
    """将分析结果渲染为 PNG 图表文件。"""

    @staticmethod
    def render(spec: ChartSpec, output_dir: str) -> Optional[str]:
        """
        渲染图表并保存为 PNG 文件。

        参数:
            spec: 图表规格
            output_dir: 输出目录（自动创建）

        返回:
            保存的文件绝对路径，失败时返回 None
        """
        if not spec.labels or not spec.values:
            logger.warning(f"[ChartGenerator] Empty data for '{spec.title}', skipping")
            return None

        chart_type = spec.chart_type.lower()
        renderers = {
            "bar": _bar_chart,
            "pie": _pie_chart,
            "line": _line_chart,
        }
        renderer = renderers.get(chart_type, _bar_chart)

        try:
            png_bytes = renderer(spec)
        except Exception as e:
            logger.error(f"[ChartGenerator] Render failed for '{spec.title}': {e}")
            return None

        os.makedirs(output_dir, exist_ok=True)
        safe_name = re.sub(r'[^\w\-]', '_', spec.title)[:40]
        filename = f"{chart_type}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, "wb") as f:
                f.write(png_bytes)
            logger.info(f"[ChartGenerator] Chart saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"[ChartGenerator] Failed to save '{filepath}': {e}")
            return None

    @staticmethod
    def from_query_result(
        raw_result: str,
        chart_type: str,
        title: str,
        x_label: str = "",
        y_label: str = "",
        output_dir: str = ".",
    ) -> Optional[str]:
        """
        直接从 SQL 查询结果字符串生成图表。

        参数:
            raw_result: SQL 查询结果字符串（LangChain 格式）
            chart_type: "bar" | "pie" | "line"
            title: 图表标题
            x_label: X 轴标签
            y_label: Y 轴标签
            output_dir: 输出目录

        返回:
            保存的文件路径，解析失败时返回 None
        """
        labels, values = _parse_query_result(raw_result)
        if not labels:
            logger.warning(f"[ChartGenerator] Could not parse result for '{title}'")
            return None

        spec = ChartSpec(
            chart_type=chart_type,
            title=title,
            labels=labels,
            values=values,
            x_label=x_label,
            y_label=y_label,
        )
        return ChartGenerator.render(spec, output_dir)
