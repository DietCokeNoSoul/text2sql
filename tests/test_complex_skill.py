"""Complex Query Skill 集成测试

测试目标：验证 ComplexQuerySkill 能正确处理需要多步规划的复杂查询，
包含多表关联聚合、排名统计等场景。

数据库：dianpng（MySQL）
表：tb_shop, tb_voucher, tb_voucher_order, tb_user, tb_blog 等
"""

import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import run_query


# ── 测试用例 ──────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "多表关联聚合：店铺优惠券统计",
        "query": (
            "统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺名称和优惠券数量"
        ),
    },
    {
        "name": "多步骤查询：用户订单分析",
        "query": (
            "查询已完成订单最多的前3名用户的ID，并查出这些用户在 tb_user 中的昵称"
        ),
    },
    {
        "name": "跨表统计：博客与关注",
        "query": (
            "分别统计 tb_blog 的博客总数、tb_follow 的关注关系总数，以及 tb_user 的用户总数，汇总成一张表"
        ),
    },
]


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

        # 判断是否经过 complex_query 路由
        routed_to_complex = "complex_query" in nodes or any(
            n in nodes for n in ["plan", "execute_step", "aggregate", "judge"]
        )

        passed = bool(final_msg) and routed_to_complex

        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status} 节点链: {' → '.join(nodes)}")
        print(f"       最终回复长度: {len(final_msg)} 字符")
        if not routed_to_complex:
            print(f"       [WARNING] 未检测到 complex_query 相关节点（可能被路由至其他 Skill）")

        return {"name": name, "passed": passed, "nodes": nodes, "error": None}

    except Exception as e:
        print(f"\n[FAIL] 异常: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "passed": False, "nodes": [], "error": str(e)}


def main():
    print("=" * 65)
    print("  Complex Query Skill — 集成测试")
    print("=" * 65)

    thread_id = f"test-complex-{uuid.uuid4().hex[:8]}"
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
        print(f"  {status}  {r['name']}{err}")

    print()
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
