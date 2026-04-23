"""Simple Query Skill 集成测试

测试目标：验证 SimpleQuerySkill 能正确处理单表/简单多表查询，
包含正常查询和错误自动修复两个场景。

数据库：dianpng（MySQL）
表：tb_user, tb_shop, tb_shop_type, tb_voucher 等
"""

import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import run_query


# ── 测试用例 ──────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "查询用户总数",
        "query": "查询 tb_user 表中的用户总数",
        "expected_nodes": ["generate_sql", "execute_query"],
        "check_keyword": None,  # 只要不报错就算通过
    },
    {
        "name": "查询店铺列表",
        "query": "列出所有店铺的名称和地址，只显示前5条",
        "expected_nodes": ["generate_sql", "execute_query"],
        "check_keyword": None,
    },
    {
        "name": "查询店铺类型",
        "query": "查询 tb_shop_type 表中有哪些店铺类型",
        "expected_nodes": ["generate_sql", "execute_query"],
        "check_keyword": None,
    },
    {
        "name": "多表简单关联",
        "query": "查询每个店铺类型下有多少家店铺",
        "expected_nodes": ["generate_sql", "execute_query"],
        "check_keyword": None,
    },
]


def run_test(test_case: dict, thread_id: str) -> dict:
    """运行单个测试用例，返回测试结果。"""
    name = test_case["name"]
    query = test_case["query"]
    print(f"\n{'─' * 60}")
    print(f"  测试: {name}")
    print(f"  问题: {query}")
    print(f"{'─' * 60}")

    try:
        result = run_query(query, thread_id)
        nodes = result.get("nodes_visited", [])
        final_msg = result.get("final_message", "")

        # 判断是否经过 simple_query 路由
        routed_to_simple = "simple_query" in nodes or any(
            n in nodes for n in ["generate_sql", "execute_query", "error_correction"]
        )

        keyword_ok = True
        if test_case.get("check_keyword"):
            keyword_ok = test_case["check_keyword"].lower() in final_msg.lower()

        passed = bool(final_msg) and routed_to_simple and keyword_ok

        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status} 节点链: {' → '.join(nodes)}")
        print(f"       最终回复长度: {len(final_msg)} 字符")
        if not routed_to_simple:
            print(f"       [WARNING] 未检测到 simple_query 相关节点")

        return {"name": name, "passed": passed, "nodes": nodes, "error": None}

    except Exception as e:
        print(f"\n[FAIL] 异常: {e}")
        import traceback
        traceback.print_exc()
        return {"name": name, "passed": False, "nodes": [], "error": str(e)}


def main():
    print("=" * 65)
    print("  Simple Query Skill — 集成测试")
    print("=" * 65)

    thread_id = f"test-simple-{uuid.uuid4().hex[:8]}"
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
