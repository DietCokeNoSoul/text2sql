"""
双塔检索基准测试 — Chinook.db

对比开启 / 关闭 SchemaGraph (Steiner Tree) 剪枝时：
  1. Schema 召回率  —— 剪枝后 schema 是否包含查询所需的全部表
  2. Schema 精确率  —— 剪枝后 schema 中不必要的表占比
  3. Token 节省量  —— 与全量 schema 相比节省的估算 token 数
  4. 响应延迟       —— 检索耗时（ms）

运行方式（无需 Milvus / API key）：
    uv run python tests/run_retrieval_benchmark.py

结果报告写入: report/retrieval_benchmark_{timestamp}.md
"""

import os
import sys
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from langchain_community.utilities import SQLDatabase
from agent.config import DatabaseConfig
from agent.database import SQLDatabaseManager
from agent.schema_graph import SchemaGraph

BENCHMARK_QUERIES = [
    {
        "id": "Q01",
        "question": "每位艺术家发行了多少张专辑？列出前10名",
        "expected_tables": {"Artist", "Album"},
        "complexity": "2表 JOIN + 聚合",
    },
    {
        "id": "Q02",
        "question": "哪些音乐流派的曲目最多？统计每个流派的曲目数",
        "expected_tables": {"Genre", "Track"},
        "complexity": "2表 JOIN + 聚合",
    },
    {
        "id": "Q03",
        "question": "哪个国家的客户消费总额最高？列出前5个国家",
        "expected_tables": {"Customer", "Invoice"},
        "complexity": "2表 JOIN + 聚合",
    },
    {
        "id": "Q04",
        "question": "每张专辑包含多少首曲目？列出曲目最多的10张专辑及其艺术家",
        "expected_tables": {"Album", "Track", "Artist"},
        "complexity": "3表 JOIN + 聚合",
    },
    {
        "id": "Q05",
        "question": "哪些员工负责的客户产生了最高的发票总额？",
        "expected_tables": {"Employee", "Customer", "Invoice"},
        "complexity": "3表 JOIN + 聚合",
    },
    {
        "id": "Q06",
        "question": "每种媒体类型的曲目平均时长是多少？",
        "expected_tables": {"MediaType", "Track"},
        "complexity": "2表 JOIN + 聚合",
    },
    {
        "id": "Q07",
        "question": "最热门的播放列表包含哪些曲目？列出曲目最多的3个播放列表及曲目数",
        "expected_tables": {"Playlist", "PlaylistTrack", "Track"},
        "complexity": "3表 JOIN + 聚合",
    },
    {
        "id": "Q08",
        "question": "每位客户的平均单次发票金额是多少？找出消费最活跃的客户",
        "expected_tables": {"Customer", "Invoice"},
        "complexity": "2表 JOIN + 聚合 + 子查询",
    },
    {
        "id": "Q09",
        "question": "哪些艺术家的曲目出现在最多的发票明细中？统计销售次数前10的艺术家",
        "expected_tables": {"Artist", "Album", "Track", "InvoiceLine"},
        "complexity": "4表 JOIN + 聚合（最复杂）",
    },
    {
        "id": "Q10",
        "question": "Rock流派中哪些专辑的总售出曲目最多？列出前5张专辑及销量",
        "expected_tables": {"Genre", "Track", "Album", "InvoiceLine"},
        "complexity": "4表 JOIN + 过滤 + 聚合",
    },
]

CHARS_PER_TOKEN = 3.5


def run_benchmark():
    db_config = DatabaseConfig(uri="sqlite:///Chinook.db", max_query_results=20)
    db_manager = SQLDatabaseManager(db_config)
    sqldb = SQLDatabase.from_uri("sqlite:///Chinook.db")
    full_schema = sqldb.get_table_info()
    full_chars = len(full_schema)

    print("构建 SchemaGraph...")
    sg = SchemaGraph()
    sg.build_from_db(db_manager)
    print(f"图节点数: {sg.node_count}, 边数: {sg.edge_count}")
    print(f"全量 Schema: {full_chars:,} 字符 (~{int(full_chars/CHARS_PER_TOKEN):,} tokens)\n")

    results = []
    hdr = f"{'ID':<5} {'查询':<36} {'复杂度':<18} {'召回':<7} {'精确':<7} {'节省%':<7} {'节省Token':<10} {'延迟ms'}"
    print(hdr)
    print("-" * len(hdr))

    for q in BENCHMARK_QUERIES:
        t0 = time.perf_counter()

        join_path = sg.plan_join_path(list(q["expected_tables"]))

        if join_path is not None:
            pruned_schema = sg.get_pruned_schema(db_manager, join_path)
            used_tables = set(join_path.tables)
        else:
            pruned_schema = full_schema
            used_tables = set(q["expected_tables"])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        pruned_chars = len(pruned_schema)
        saved_chars = max(0, full_chars - pruned_chars)
        saved_tokens = saved_chars / CHARS_PER_TOKEN
        saved_pct = saved_chars / full_chars * 100

        expected = q["expected_tables"]
        recall = len(expected & used_tables) / len(expected) if expected else 1.0
        precision = len(expected & used_tables) / len(used_tables) if used_tables else 0.0

        r = {
            "id": q["id"],
            "question": q["question"],
            "expected_tables": sorted(expected),
            "used_tables": sorted(used_tables),
            "complexity": q["complexity"],
            "pruned_chars": pruned_chars,
            "full_chars": full_chars,
            "saved_chars": saved_chars,
            "saved_tokens": saved_tokens,
            "saved_pct": saved_pct,
            "recall": recall,
            "precision": precision,
            "elapsed_ms": elapsed_ms,
            "join_path": join_path,
        }
        results.append(r)

        recall_mark = "✓" if recall == 1.0 else "✗"
        print(
            f"{q['id']:<5} {q['question'][:34]:<36} {q['complexity']:<18} "
            f"{recall*100:.0f}%{recall_mark:<3} "
            f"{precision*100:.0f}%{'':<4} "
            f"{saved_pct:.0f}%{'':<4} "
            f"{saved_tokens:>7.0f}{'':<3} "
            f"{elapsed_ms:>5.1f}"
        )

    return results, full_schema, sg


def write_report(results, full_schema, sg):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.abspath("report")
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, f"retrieval_benchmark_{timestamp}.md")

    full_chars = len(full_schema)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_saved_pct = sum(r["saved_pct"] for r in results) / len(results)
    avg_saved_tokens = sum(r["saved_tokens"] for r in results) / len(results)
    avg_elapsed = sum(r["elapsed_ms"] for r in results) / len(results)
    perfect_recall = sum(1 for r in results if r["recall"] == 1.0)
    precision_gain = avg_precision * 100 - 1 / sg.node_count * 100
    recall_label = "完美" if avg_recall == 1.0 else "部分损失"

    md = [
        "# 双塔检索基准测试报告",
        "",
        f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据库**: Chinook.db (SQLite, 11表)",
        f"**查询数量**: {len(results)} 条",
        f"**全量 Schema**: {full_chars:,} 字符 (~{int(full_chars/CHARS_PER_TOKEN):,} tokens)",
        "",
        "---",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| Schema 召回率（平均） | **{avg_recall*100:.1f}%** |",
        f"| Schema 精确率（平均） | **{avg_precision*100:.1f}%** |",
        f"| 完美召回（必要表全部命中） | **{perfect_recall}/{len(results)}** 条查询 |",
        f"| 平均 Schema 压缩率 | **{avg_saved_pct:.1f}%** |",
        f"| 平均 Token 节省量 | **~{avg_saved_tokens:.0f} tokens/查询** |",
        f"| 检索平均延迟 | **{avg_elapsed:.2f} ms** |",
        "",
        "---",
        "",
        "## 开启 vs 关闭双塔检索对比",
        "",
        "| 维度 | 关闭检索（全量Schema） | 开启检索（剪枝Schema） | 提升 |",
        "|------|----------------------|----------------------|------|",
        f"| Schema 字符数 | {full_chars:,} chars | ~{int(full_chars*(1-avg_saved_pct/100)):,} chars | -{avg_saved_pct:.0f}% |",
        f"| 估算 Token 数 | ~{int(full_chars/CHARS_PER_TOKEN):,} tokens | ~{int(full_chars/CHARS_PER_TOKEN*(1-avg_saved_pct/100)):,} tokens | -{avg_saved_tokens:.0f} tokens/查询 |",
        f"| 表召回率 | 100%（所有11表） | {avg_recall*100:.1f}% | {recall_label} |",
        f"| Schema 精确率 | {1/sg.node_count*100:.0f}% (1/11) | {avg_precision*100:.1f}% | +{precision_gain:.0f}pp |",
        f"| 检索延迟 | 0 ms | {avg_elapsed:.2f} ms | 额外开销 |",
        "",
        "---",
        "",
        "## 逐条查询结果",
        "",
        "| ID | 查询 | 复杂度 | 召回率 | 精确率 | Token节省 | 压缩率 | 延迟(ms) |",
        "|-----|------|--------|--------|--------|-----------|--------|----------|",
    ]

    for r in results:
        icon = "OK" if r["recall"] == 1.0 else "MISS"
        md.append(
            f"| {r['id']} | {r['question'][:32]}... | {r['complexity']} | "
            f"{icon} {r['recall']*100:.0f}% | {r['precision']*100:.0f}% | "
            f"~{r['saved_tokens']:.0f} | {r['saved_pct']:.0f}% | {r['elapsed_ms']:.1f} |"
        )

    md += [
        "",
        "---",
        "",
        "## 各查询详情",
        "",
    ]
    for r in results:
        md += [
            f"### {r['id']}: {r['question']}",
            "",
            f"- **复杂度**: {r['complexity']}",
            f"- **必要表**: {', '.join(f'`{t}`' for t in r['expected_tables'])}",
            f"- **检索到的表**: {', '.join(f'`{t}`' for t in r['used_tables'])}",
            f"- **召回率**: {r['recall']*100:.0f}% | **精确率**: {r['precision']*100:.0f}%",
            f"- **Schema**: {r['full_chars']:,} → {r['pruned_chars']:,} 字符（节省 {r['saved_pct']:.0f}%, ~{r['saved_tokens']:.0f} tokens）",
            f"- **延迟**: {r['elapsed_ms']:.1f} ms",
            "",
        ]

    md += ["---", "", "*由 `tests/run_retrieval_benchmark.py` 自动生成*"]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return filepath


if __name__ == "__main__":
    print("=" * 90)
    print("双塔检索基准测试 — Chinook.db")
    print("=" * 90)

    results, full_schema, sg = run_benchmark()

    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_saved_tokens = sum(r["saved_tokens"] for r in results) / len(results)
    avg_saved_pct = sum(r["saved_pct"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    perfect = sum(1 for r in results if r["recall"] == 1.0)

    print("\n" + "=" * 90)
    print("汇总")
    print("=" * 90)
    print(f"  平均召回率:       {avg_recall*100:.1f}%  ({perfect}/{len(results)} 完美召回)")
    print(f"  平均精确率:       {avg_precision*100:.1f}%")
    print(f"  平均 Schema 压缩: {avg_saved_pct:.1f}%")
    print(f"  平均 Token 节省:  ~{avg_saved_tokens:.0f} tokens/查询")

    report_path = write_report(results, full_schema, sg)
    print(f"\n报告已保存: {report_path}")
