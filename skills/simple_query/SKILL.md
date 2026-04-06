# Simple Query Skill

## 描述
处理单表或简单多表 SQL 查询的 Skill。

## 能力
- 📋 列出数据库表
- 🔍 获取相关表结构
- ✍️ 生成 SQL 查询
- ✅ 验证查询正确性
- ▶️ 执行查询并返回结果

## 输入
```yaml
messages: 用户的查询请求
  - "查询所有客户"
  - "统计订单总数"
  - "查找价格最高的产品"
```

## 输出
```yaml
messages: 包含查询结果的消息历史
query_result: SQL 查询结果
```

## 依赖
- LLM: 用于生成和验证 SQL
- SQLDatabaseToolkit: 数据库操作
- CommonNodes: 复用公共节点 (list_tables, get_schema, execute_query)

## 流程
```
START
  ↓
list_tables (复用)
  ↓
get_schema (复用)
  ↓
generate_query (特有节点)
  ↓
check_query (特有节点)
  ↓
run_query (复用)
  ↓
END
```

## 示例
```python
from skills.simple_query import SimpleQuerySkill

skill = SimpleQuerySkill(llm, tool_manager, db_manager)
result = skill.invoke({
    "messages": [HumanMessage(content="查询所有客户")]
})
print(result["messages"][-1].content)
```

## 适用场景
- ✅ 单表查询
- ✅ 简单的多表 JOIN
- ✅ 聚合统计 (COUNT, SUM, AVG)
- ❌ 复杂的子查询（使用 Complex Query Skill）
- ❌ 多步骤分析（使用 Data Analysis Skill）
