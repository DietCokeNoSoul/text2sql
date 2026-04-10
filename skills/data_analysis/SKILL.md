# Data Analysis Skill

**Comprehensive 7-step data analysis with insights and visualization**

## Purpose
Performs end-to-end data analysis including exploration, insight extraction, and visualization recommendations. Suitable for business intelligence, trend analysis, and data-driven decision making.

## Architecture

### Flow
```
understand_goal → explore_data → plan_analysis → 
generate_queries → analyze_results → visualize → generate_report
```

### Nodes

1. **understand_goal**: Extract analysis objectives, metrics, and requirements
2. **explore_data**: Explore database structure and gather statistics
3. **plan_analysis**: Create detailed analysis plan with steps
4. **generate_queries**: Generate optimized SQL for each analysis step
5. **analyze_results**: Execute queries and extract insights
6. **visualize**: Generate visualization recommendations
7. **generate_report**: Create comprehensive analysis report

## Features
- ✅ Automatic analysis planning
- ✅ Multi-step query execution
- ✅ Insight extraction from results
- ✅ Visualization recommendations
- ✅ Comprehensive report generation
- ✅ Business intelligence focus

## Usage

```python
from skills.data_analysis import DataAnalysisSkill

skill = DataAnalysisSkill(llm, tool_manager, db_manager)
result = skill.invoke({
    "messages": [HumanMessage(content="Analyze user engagement trends")]
})

# Access the report
report = result.get("report")
print(report)
```

## When to Use
- Business intelligence queries
- Trend analysis
- Performance metrics analysis
- Data exploration with insights
- Questions requiring interpretation and recommendations

## Output

The skill generates:
- **Insights**: Key findings from data
- **Visualizations**: Chart recommendations
- **Report**: Comprehensive analysis document
