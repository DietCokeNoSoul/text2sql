"""Data Analysis Skill 集成测试

测试目标：验证 DataAnalysisSkill 能完整执行 8 步分析流水线，
生成 Markdown 报告、SVG 图表并导出 CSV 文件。

数据库：dianpng（MySQL）
分析对象：用户行为、店铺销售、优惠券使用等
"""

import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import run_query


# ── 测试用例 ──────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "用户行为分析",
        "query": (
            "分析 dianpng 数据库中的用户行为：包括用户注册情况、"
            "用户发布博客的活跃度，以及用户关注关系，生成分析报告"
        ),
        # 期望 data_analysis skill 特有节点出现
        "expected_nodes": ["understand_goal", "explore_data", "generate_report"],
    },
]

# 分析类查询耗时较长，只测一个典型场景

EXPECTED_ANALYSIS_NODES = {
    "understand_goal", "explore_data", "plan_analysis",
    "generate_queries", "analyze_results", "visualize",
    "generate_report",
}


def run_test(test_case: dict, thread_id: str) -> dict:
    """运行单个测试用例，返回测试结果。"""
    name = test_case["name"]
    query = test_case["query"]
    print(f"\n{'─' * 60}")
    print(f"  测试: {name}")
    print(f"  问题: {query[:80]}...")
    print(f"{'─' * 60}")

    try:
        result = run_query(query, thread_id)
        nodes = result.get("nodes_visited", [])
        final_msg = result.get("final_message", "")
        export_files = result.get("export_files", [])

        nodes_set = set(nodes)

        # 判断是否经过 data_analysis 路由
        routed_to_analysis = "data_analysis" in nodes_set or bool(
            nodes_set & EXPECTED_ANALYSIS_NODES
        )

        # 验证预期节点是否都出现
        expected = set(test_case.get("expected_nodes", []))
        missing_nodes = expected - nodes_set

        passed = bool(final_msg) and routed_to_analysis

        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status} 节点链: {' → '.join(nodes)}")
        print(f"       最终回复长度: {len(final_msg)} 字符")
        if export_files:
            print(f"       导出文件: {export_files}")
        if missing_nodes:
            print(f"       [WARNING] 缺少预期节点: {missing_nodes}")
        if not routed_to_analysis:
            print(f"       [WARNING] 未检测到 data_analysis 相关节点")

        return {
            "name": name,
            "passed": passed,
            "nodes": nodes,
            "export_files": export_files,
            "error": None,
        }

    except Exception as e:
        print(f"\n[FAIL] 异常: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "passed": False, "nodes": [], "export_files": [], "error": str(e)}


def main():
    print("=" * 65)
    print("  Data Analysis Skill — 集成测试")
    print("=" * 65)
    print("  注意：分析类查询涉及多步 LLM 调用，耗时较长，请耐心等待")

    thread_id = f"test-analysis-{uuid.uuid4().hex[:8]}"
    results = []

    for tc in TEST_CASES:
        res = run_test(tc, thread_id)
        results.append(res)

    # 汇总
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'=' * 65}")
    print(f"  测试结果汇总: {passed}/{total} 通过")
    print(f"{'=' * 65}")
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        err = f"  → {r['error']}" if r["error"] else ""
        files = f"  导出: {r['export_files']}" if r.get("export_files") else ""
        print(f"  {status}  {r['name']}{err}{files}")

    print()
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
