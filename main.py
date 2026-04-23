"""
Text2SQL LangGraph Agent — 主入口

使用方式::

    python main.py            # 启动交互式命令行问答循环

run_query / run_query_streaming 返回结构::

    {
        "final_message": str,        # LLM 最终回复文本
        "nodes_visited": list[str],  # 图中依次经过的节点名称
        "export_files":  list[str],  # DataAnalysis 模式下导出的 CSV/Excel 路径
    }

三种技能路由::

    简单查询  — 单轮 SQL 生成 + 执行 + 自我修正
    复杂查询  — Plan-Execute 多步分解，可选双塔检索剪枝 schema
    数据分析  — 8 步深度分析：目标理解 → 探索 → 规划 → 生成 SQL →
                执行 → 洞察 → 报告 → 导出 CSV/Excel
"""

from agent.graph import main

if __name__ == "__main__":
    main()