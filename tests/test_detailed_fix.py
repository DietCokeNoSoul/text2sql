"""详细测试 - 显示完整的消息历史和 SQL 语句"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入项目配置和 LLM
from agent.config import get_config
from langchain_community.chat_models import ChatTongyi
from agent.database import SQLDatabaseManager
from agent.tools import SQLToolManager
from skills.simple_query.skill import SimpleQuerySkill

def print_message_history(messages):
    """打印完整的消息历史"""
    print("\n" + "="*60)
    print("消息历史:")
    print("="*60)
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        # 提取 tool_calls 信息
        tool_info = ""
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get('name', 'unknown')
                tool_args = tc.get('args', {})
                # 提取 SQL 查询
                query = tool_args.get('query', tool_args.get('sql', ''))
                if query:
                    tool_info = f"\n      工具: {tool_name}\n      SQL: {query}"
                else:
                    tool_info = f"\n      工具: {tool_name}"
        
        print(f"\n[{i+1}] {msg_type}:")
        print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")
        if tool_info:
            print(f"    {tool_info}")
    print("="*60 + "\n")

def test_single_query(skill, query, test_name):
    """测试单个查询并显示详细信息"""
    print("\n" + "="*80)
    print(f"测试: {test_name}")
    print(f"查询: {query}")
    print("="*80)
    
    result = skill.invoke({'messages': [('user', query)]})
    
    # 显示消息历史
    print_message_history(result['messages'])
    
    # 显示最终结果
    print(f"最终结果: {result['messages'][-1].content}")
    print(f"重试次数: {result.get('retry_count', 0)}")
    print(f"最后错误: {result.get('last_error', 'None')}")
    print(f"最后 SQL: {result.get('last_sql', 'None')}")
    
    return result

def main():
    # 获取项目配置
    config = get_config()
    
    # 初始化 LLM
    print(f"正在初始化 LLM: {config.llm.provider} - {config.llm.model}...")
    llm = ChatTongyi(
        model=config.llm.model,
        temperature=config.llm.temperature,
        dashscope_api_key=config.llm.api_key
    )
    
    print("正在连接数据库...")
    db_manager = SQLDatabaseManager(config.database)
    
    print("正在初始化工具管理器...")
    tool_manager = SQLToolManager(db_manager, llm)
    
    print("正在初始化 SimpleQuerySkill...")
    skill = SimpleQuerySkill(llm, tool_manager, db_manager)
    print("初始化完成！\n")
    
    # 只测试最关键的列名错误修复场景
    test_single_query(
        skill,
        "查询商店的店名和评分，只显示前3个",
        "列名错误修复 (shop_name → name)"
    )

if __name__ == "__main__":
    main()
