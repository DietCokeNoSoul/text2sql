"""测试 Complex Query Skill - 详细版本"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import get_config
from langchain_community.chat_models import ChatTongyi
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from skills.complex_query.skill import ComplexQuerySkill

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def main():
    # 初始化
    print("初始化组件...")
    config = get_config()
    llm = ChatTongyi(
        model=config.llm.model,
        temperature=config.llm.temperature,
        dashscope_api_key=config.llm.api_key
    )
    db_manager = SQLDatabaseManager(config.database)
    tool_manager = SQLToolManager(db_manager, llm)
    skill = ComplexQuerySkill(llm, tool_manager, db_manager)
    print("✓ 初始化完成\n")
    
    # 测试用例 - 使用明确需要多步骤的查询
    print_section("测试: 复杂多步骤查询")
    question = """
    分析商店数据：
    1. 先找出评分最高的前3个商店类型
    2. 然后对于这3个类型，分别统计每个类型下有多少家商店
    3. 最后列出每个类型下评分最高的商店名称
    请分步骤执行，并汇总结果
    """
    print(f"问题: {question.strip()}\n")
    
    try:
        result = skill.invoke({'messages': [('user', question)]})
        
        # 显示查询计划
        query_plan = result.get('query_plan', [])
        print(f"\n查询计划（{len(query_plan)} 个步骤）:")
        for step in query_plan:
            print(f"\n  步骤 {step['step_id']}: {step['description']}")
            print(f"    SQL: {step.get('query', 'N/A')}")
            print(f"    依赖: {step.get('depends_on', [])}")
        
        # 显示执行结果
        step_results = result.get('step_results', {})
        print(f"\n执行结果（{len(step_results)} 个完成）:")
        for step_id, res in sorted(step_results.items()):
            status = "✓" if res.get('success') else "✗"
            print(f"\n  {status} 步骤 {step_id}: {res.get('description', 'N/A')}")
            
            # 显示原始查询和实际执行的查询
            if res.get('original_query') and res.get('original_query') != res.get('query'):
                print(f"    原始SQL: {res.get('original_query', 'N/A')}")
                print(f"    实际SQL: {res.get('query', 'N/A')}")
            else:
                print(f"    SQL: {res.get('query', 'N/A')}")
            
            if res.get('success'):
                result_str = str(res.get('result', ''))
                if result_str:
                    print(f"    结果: {result_str[:300]}{'...' if len(result_str) > 300 else ''}")
                else:
                    print(f"    结果: (空)")
            else:
                print(f"    错误: {res.get('error', 'Unknown')}")
        
        # 显示最终消息
        messages = result.get('messages', [])
        if messages:
            print("\n最终回复:")
            for msg in messages[-3:]:  # 只显示最后3条消息
                content = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")
        
        print(f"\n状态: 计划完成={result.get('plan_completed', False)}")
        print("\n✓ 测试完成")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()