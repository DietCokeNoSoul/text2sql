# Data Analysis Skill

**执行端到端 8 步数据分析流程，含洞察发现、可视化建议和结果导出**

## 目的

对用户的分析类问题进行完整的数据探索、多维度查询、洞察提炼、图表建议和报告生成，适合商业智能和数据驱动决策场景。

## 适用场景

- 趋势分析（时间维度的变化规律）
- 商业智能查询（销售、用户、运营指标分析）
- 数据探索（发现规律、异常、相关性）
- 需要洞察解读和建议的问题
- 需要生成可视化图表或分析报告

## 不适用场景

- 直接取数的简单查询（请使用 Simple Query Skill）
- 多步骤但无需分析解读的查询（请使用 Complex Query Skill）

## 流程

```
understand_goal → explore_data → plan_analysis →
generate_queries → analyze_results → visualize → generate_report → export_results
```

## 能力

- 🎯 分析目标理解（understand_goal）
- 🔍 数据库结构探索（explore_data）
- 📋 自动生成分析计划（plan_analysis）
- 🛠️ 多步骤 SQL 查询生成与执行
- 💡 数据洞察提炼
- 📈 可视化图表建议与生成（PNG）
- 📝 完整中文 Markdown 分析报告
- 📤 查询结果导出（CSV/Excel）

## 示例

```python
from skills.data_analysis import DataAnalysisSkill

skill = DataAnalysisSkill(llm, tool_manager, db_manager)
result = skill.invoke({
    "messages": [HumanMessage(content="分析过去三个月的用户活跃度趋势")]
})
report = result.get("report")
print(report)
```

