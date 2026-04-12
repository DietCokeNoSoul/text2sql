"""
Test Data Analysis Skill - Detailed Version

Tests the 7-step comprehensive data analysis workflow with detailed logging.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

from agent.config import get_config
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from skills.data_analysis import DataAnalysisSkill

def main():
    """Test data analysis skill"""
    
    print("="*80)
    print("  DATA ANALYSIS SKILL - 详细测试")
    print("="*80)
    
    # Initialize components
    print("\n初始化组件...")
    try:
        config = get_config()
        llm = ChatTongyi(model=config.llm.model, dashscope_api_key=config.llm.api_key)
        db_manager = SQLDatabaseManager(config.database)
        tool_manager = SQLToolManager(db_manager, llm)
        skill = DataAnalysisSkill(llm, tool_manager, db_manager)
        print("✓ 初始化完成")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test query
    test_query = """
    分析商店数据：
    1. 有多少家商店？
    2. 每个类型有多少家商店？
    3. 哪些商店评分最高？
    4. 评分分布如何？
    
    请提供洞察和可视化建议。
    """
    
    print("\n" + "="*80)
    print("  测试查询")
    print("="*80)
    print(f"问题: {test_query.strip()}")
    
    print("\n开始执行 7 步分析流程...\n")
    
    initial_state = {"messages": [HumanMessage(content=test_query)]}
    
    try:
        result = skill.invoke(initial_state)
        
        print("\n" + "="*80)
        print("  执行结果")
        print("="*80)
        
        # 1. Analysis Goal
        analysis_goal = result.get("analysis_goal", "")
        if analysis_goal:
            print("\n📋 步骤 1: 理解目标")
            print(f"  {analysis_goal[:300]}")
            if len(analysis_goal) > 300:
                print("  ...")
        
        # 2. Tables explored
        tables = result.get("tables", [])
        if tables:
            print(f"\n📊 步骤 2: 探索数据")
            print(f"  发现 {len(tables)} 个表: {', '.join(tables[:5])}")
        
        # 3. Analysis Plan
        analysis_plan = result.get("analysis_plan", {})
        if analysis_plan:
            steps = analysis_plan.get("steps", [])
            print(f"\n📝 步骤 3: 分析计划 ({len(steps)} 个步骤)")
            for i, step in enumerate(steps[:5], 1):
                desc = step.get("description", "N/A")
                print(f"  {i}. {desc[:80]}")
        
        # 4. SQL Queries
        sql_queries = result.get("sql_queries", [])
        if sql_queries:
            print(f"\n🔍 步骤 4: 生成查询 ({len(sql_queries)} 个查询)")
            for query in sql_queries[:3]:
                step_id = query.get("step_id", "?")
                sql = query.get("query", "N/A")
                print(f"  步骤 {step_id}: {sql[:100]}")
                if len(sql) > 100:
                    print("    ...")
        
        # 5. Query Results
        query_results = result.get("query_results", [])
        if query_results:
            successful = sum(1 for r in query_results if r.get("success"))
            failed = len(query_results) - successful
            print(f"\n✓ 步骤 5: 分析结果 ({successful} 成功, {failed} 失败)")
            
            # Show sample results
            for i, res in enumerate(query_results[:3], 1):
                step_id = res.get("step_id", "?")
                success = res.get("success", False)
                status = "✓" if success else "✗"
                print(f"  {status} 步骤 {step_id}:", end=" ")
                
                if success:
                    result_data = str(res.get("result", ""))[:80]
                    print(result_data)
                else:
                    error = res.get("error", "Unknown error")[:80]
                    print(f"Error: {error}")
        
        # 6. Insights
        insights = result.get("insights", [])
        if insights:
            print(f"\n💡 步骤 6: 洞察提取 ({len(insights)} 个洞察)")
            for i, insight in enumerate(insights[:3], 1):
                text = insight.get("insight", "N/A")[:150]
                print(f"  {i}. {text}")
                if len(insight.get("insight", "")) > 150:
                    print("     ...")
        
        # 7. Visualizations
        visualizations = result.get("visualizations", [])
        if visualizations:
            print(f"\n📊 步骤 7: 可视化建议 ({len(visualizations)} 个图表)")
            for i, viz in enumerate(visualizations[:3], 1):
                chart_type = viz.get("chart_type", "N/A")
                title = viz.get("title", "N/A")
                print(f"  {i}. {chart_type}: {title}")
        
        # 8. Final Report
        report = result.get("report", "")
        if report:
            print(f"\n📄 最终报告 ({len(report)} 字符)")
            print("\n" + "="*80)
            print("  报告预览")
            print("="*80)
            lines = report.split('\n')
            preview_lines = lines[:20]
            for line in preview_lines:
                print(line)
            if len(lines) > 20:
                print(f"\n... (还有 {len(lines) - 20} 行)")
            print("="*80)
        
        # Final messages
        messages = result.get("messages", [])
        print(f"\n消息数量: {len(messages)}")
        
        print("\n" + "="*80)
        print("✓ 测试完成 - Analysis Skill 执行成功!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ 测试失败: {type(e).__name__}: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()