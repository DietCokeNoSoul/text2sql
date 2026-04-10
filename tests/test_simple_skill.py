"""测试 Simple Query Skill 的独立运行。

此脚本验证 Simple Query Skill 能够正常工作。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.chat_models import ChatTongyi
from langchain.messages import HumanMessage

from agent.config import get_config
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from skills.simple_query import SimpleQuerySkill


def test_simple_query_skill():
    """测试 Simple Query Skill。"""
    
    print("=" * 60)
    print("Simple Query Skill Integration Test")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n[1/5] Loading configuration...")
    config = get_config()
    print(f"  OK Database: {config.database.uri}")
    print(f"  OK LLM: {config.llm.provider}/{config.llm.model}")
    
    # 2. 初始化组件
    print("\n[2/5] Initializing components...")
    
    # LLM
    llm = ChatTongyi(
        model=config.llm.model,
        dashscope_api_key=config.llm.api_key
    )
    print("  OK LLM initialized")
    
    # 数据库管理器
    db_manager = SQLDatabaseManager(config.database)
    print("  OK Database manager initialized")
    
    # 工具管理器
    tool_manager = SQLToolManager(db_manager, llm)  # 修复：传入 db_manager 对象
    print("  OK Tool manager initialized")
    
    # 3. 创建 Skill
    print("\n[3/5] Creating Simple Query Skill...")
    skill = SimpleQuerySkill(llm, tool_manager, db_manager)
    print(f"  OK Skill name: {skill.name}")
    print(f"  OK Skill description: {skill.description}")
    
    # 4. 准备测试查询
    print("\n[4/5] Preparing test queries...")
    test_queries = [
        "List all tables",
        "Query first 5 customers",
        "Count total orders"
    ]
    
    print(f"  Prepared {len(test_queries)} test queries")
    
    # 5. 执行测试
    print("\n[5/5] Running tests...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-' * 60}")
        print(f"Test {i}/{len(test_queries)}: {query}")
        print('-' * 60)
        
        try:
            # 准备输入状态
            from langchain.messages import HumanMessage
            
            initial_state = {
                "messages": [HumanMessage(content=query)]
            }
            
            print(f"  Input: {query}")
            print(f"  Executing...")
            
            # 执行 Skill
            result = skill.invoke(initial_state)
            
            # 显示结果
            print(f"\n  OK Execution successful!")
            print(f"  Message count: {len(result.get('messages', []))}")
            
            # 显示最后几条消息
            messages = result.get('messages', [])
            if messages:
                print(f"\n  Last messages:")
                for msg in messages[-3:]:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    # 截断长内容
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"    - {content}")
            
        except Exception as e:
            print(f"\n  FAIL Test failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 继续测试下一个
            continue
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)


if __name__ == "__main__":
    test_simple_query_skill()
