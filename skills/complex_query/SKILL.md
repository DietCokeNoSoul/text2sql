# Complex Query Skill

**Plan-Execute pattern with parallel execution for multi-step queries**

## Purpose
Handles complex SQL queries that require breaking down into multiple sub-queries. Uses LangGraph's Send API to execute independent steps in parallel.

## Architecture

### Flow
```
list_tables → get_schema → plan → execute_steps (parallel) → aggregate → judge → [loop or end]
```

### Nodes
1. **list_tables** (common): List all available tables
2. **get_schema** (common): Get schema for relevant tables
3. **plan**: Analyze question, generate multi-step query plan
4. **execute_step**: Execute query steps (parallel via Send API)
5. **aggregate**: Collect and format all step results
6. **judge**: Check if all steps completed, decide continue/end

## Features
- ✅ Automatic query decomposition
- ✅ Parallel step execution (Send API)
- ✅ Dependency tracking between steps
- ✅ Automatic retry for failed steps
- ✅ Result aggregation

## Usage

```python
from skills.complex_query import ComplexQuerySkill

skill = ComplexQuerySkill(llm, tool_manager, db_manager)
result = skill.invoke({
    "messages": [HumanMessage(content="Find top users and their comments")]
})
```

## When to Use
- Multi-table queries requiring joins
- Questions needing intermediate results
- Analysis requiring multiple data points
- Queries with dependencies between steps
