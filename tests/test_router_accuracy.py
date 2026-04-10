"""
主图路由准确性测试

只测试 query_router 的分类准确性，不执行实际 Skill。
"""

import sys
sys.path.insert(0, r"c:\Users\71949\Desktop\text2sql")

from langchain_community.chat_models import ChatTongyi
from langchain.messages import HumanMessage

from agent.config import get_config
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from agent.skill_graph_builder import SkillBasedGraphBuilder


# 测试用例矩阵：覆盖三种类型，每种多个场景
TEST_CASES = [
    # === SIMPLE 查询 ===
    {"query": "查询所有用户", "expected": "simple", "category": "简单查询"},
    {"query": "显示前10个商店", "expected": "simple", "category": "简单查询"},
    {"query": "查找 tb_shop 表中 id=5 的商店", "expected": "simple", "category": "简单查询"},
    {"query": "统计用户总数", "expected": "simple", "category": "简单查询"},
    {"query": "查询评分大于45的商店名称和评分", "expected": "simple", "category": "简单查询"},
    
    # === COMPLEX 查询 ===
    {"query": "找出每个商店类型中评分最高的前3个商店，并统计每个类型的平均评分", "expected": "complex", "category": "复杂查询"},
    {"query": "查询拥有博客数量最多的前5个用户，并显示他们的博客评论总数", "expected": "complex", "category": "复杂查询"},
    {"query": "先找出评分最高的3个商店类型，然后统计每个类型下的商店数量", "expected": "complex", "category": "复杂查询"},
    {"query": "比较不同类型商店的平均评分、最高评分和商店数量", "expected": "complex", "category": "复杂查询"},
    {"query": "分步骤执行：第一步找出所有有秒杀券的商店，第二步统计每个商店的秒杀券数量", "expected": "complex", "category": "复杂查询"},
    
    # === ANALYSIS 查询 ===
    {"query": "分析商店数据，提供洞察和可视化建议", "expected": "analysis", "category": "分析查询"},
    {"query": "分析用户参与度趋势，哪些用户最活跃，给出改进建议", "expected": "analysis", "category": "分析查询"},
    {"query": "生成一份商店类型分布和评分分析报告", "expected": "analysis", "category": "分析查询"},
    {"query": "深入分析平台数据，找出影响商店评分的关键因素，并给出优化建议", "expected": "analysis", "category": "分析查询"},
    {"query": "对博客数据做全面分析，包括发布趋势、用户参与度、热门话题，并生成可视化报告", "expected": "analysis", "category": "分析查询"},
]


def main():
    print("=" * 80)
    print("  主图路由准确性测试")
    print("=" * 80)
    
    # 初始化
    print("\n初始化组件...")
    try:
        config = get_config()
        llm = ChatTongyi(model=config.llm.model, dashscope_api_key=config.llm.api_key)
        db_manager = SQLDatabaseManager(config.database)
        tool_manager = SQLToolManager(db_manager, llm)
        builder = SkillBasedGraphBuilder(
            config=config, llm=llm, db_manager=db_manager, tool_manager=tool_manager
        )
        print("✓ 初始化完成\n")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 执行测试
    results = []
    total = len(TEST_CASES)
    
    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expected = case["expected"]
        category = case["category"]
        
        print(f"[{i}/{total}] {category}")
        print(f"  问题: {query[:60]}{'...' if len(query) > 60 else ''}")
        
        try:
            # 只调用路由节点，不执行 Skill
            state = {"messages": [HumanMessage(content=query)]}
            router_result = builder._query_router_node(state)
            actual = router_result.get("query_type", "unknown")
            
            correct = actual == expected
            icon = "✓" if correct else "✗"
            print(f"  {icon} 预期: {expected}, 实际: {actual}")
            
            results.append({
                "query": query,
                "category": category,
                "expected": expected,
                "actual": actual,
                "correct": correct,
            })
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results.append({
                "query": query,
                "category": category,
                "expected": expected,
                "actual": "error",
                "correct": False,
            })
        
        print()
    
    # === 汇总 ===
    print("=" * 80)
    print("  测试结果汇总")
    print("=" * 80)
    
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total * 100
    
    print(f"\n总体准确率: {correct_count}/{total} ({accuracy:.0f}%)\n")
    
    # 按类别统计
    for cat in ["简单查询", "复杂查询", "分析查询"]:
        cat_results = [r for r in results if r["category"] == cat]
        cat_correct = sum(1 for r in cat_results if r["correct"])
        cat_total = len(cat_results)
        cat_pct = cat_correct / cat_total * 100 if cat_total > 0 else 0
        icon = "✓" if cat_correct == cat_total else "⚠"
        print(f"  {icon} {cat}: {cat_correct}/{cat_total} ({cat_pct:.0f}%)")
    
    # 显示错误的案例
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n错误分类 ({len(wrong)} 个):")
        for r in wrong:
            print(f"  ✗ \"{r['query'][:50]}...\"")
            print(f"    预期: {r['expected']}, 实际: {r['actual']}")
    else:
        print("\n🎉 所有查询分类正确！")
    
    print("\n" + "=" * 80)
    if accuracy == 100:
        print("✓ 路由测试通过 - 准确率 100%")
    elif accuracy >= 80:
        print(f"⚠ 路由测试基本通过 - 准确率 {accuracy:.0f}%")
    else:
        print(f"✗ 路由测试不通过 - 准确率 {accuracy:.0f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
