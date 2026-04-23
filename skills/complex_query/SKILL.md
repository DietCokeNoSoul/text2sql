# Complex Query Skill

**处理需要分解为多步骤执行的复杂 SQL 查询（Plan-Execute 模式）**

## 目的

将复杂问题自动分解为多个子查询步骤，并行执行后聚合结果，支持步骤间数据依赖（Placeholder 机制）。

## 适用场景

- 需要跨多表联合分析的查询
- 问题需要中间结果才能继续（如先查 ID 再查详情）
- 含排名、对比、多维度交叉分析
- 步骤之间有数据依赖关系

## 不适用场景

- 单表简单过滤查询（请使用 Simple Query Skill）
- 需要趋势洞察、可视化或分析报告（请使用 Data Analysis Skill）

## 流程

```
list_tables → get_schema → plan → execute_steps（并行） → aggregate → judge → [循环或结束]
```

## 能力

- 🗂️ 自动查询分解（Plan 节点）
- ⚡ 步骤并行执行（LangGraph Send API）
- 🔗 步骤间依赖解析（Placeholder 机制）
- 🔄 失败步骤自动重试
- 📊 多步骤结果聚合

## 示例

```python
from skills.complex_query import ComplexQuerySkill

skill = ComplexQuerySkill(llm, tool_manager, db_manager)
result = skill.invoke({
    "messages": [HumanMessage(content="找出每个分类中销量最高的前3个商品及其店铺信息")]
})
```

