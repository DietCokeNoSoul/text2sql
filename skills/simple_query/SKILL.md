# Simple Query Skill

**处理单表或简单多表 SQL 查询的 Skill**

## 目的

处理可以用单条或少量 SQL 直接回答的查询问题，包含自动错误修复和列名模糊匹配。

## 适用场景

- 单表查询（列出、过滤、排序）
- 简单的多表 JOIN
- 聚合统计（COUNT / SUM / AVG / MAX / MIN）
- 按条件筛选记录

## 不适用场景

- 需要分解为多个步骤的复杂查询（请使用 Complex Query Skill）
- 需要趋势分析、洞察发现或可视化的问题（请使用 Data Analysis Skill）

## 流程

```
list_tables → get_schema → generate_query → execute_query
                                              ↓ 失败（最多3次）
                                           fix_query → execute_query
```

## 能力

- 📋 自动列出数据库表
- 🔍 获取相关表结构
- ✍️ 生成 SQL 查询
- 🔧 自动修复错误 SQL（最多重试3次）
- 🔎 列名模糊匹配（自动建议相似列名）
- ▶️ 执行查询并返回结果

## 示例

```python
from skills.simple_query import SimpleQuerySkill

skill = SimpleQuerySkill(llm, tool_manager, db_manager)
result = skill.invoke({
    "messages": [HumanMessage(content="查询所有客户")]
})
print(result["messages"][-1].content)
```

