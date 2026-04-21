"""Text2SQL Agent 测试套件。

测试文件说明:
  test_simple_skill.py        - Simple Query Skill 端到端测试（需 API Key）
  test_complex_detailed.py    - Complex Query Skill 详细测试（需 API Key）
  test_analysis_detailed.py   - Data Analysis Skill 详细测试（需 API Key）
  test_router_accuracy.py     - 主图路由准确率测试，15 个用例 100%（需 API Key）
  test_main_graph.py          - 主图集成测试（需 API Key）
  test_detailed_fix.py        - SQL 错误修复详细测试（需 API Key）
  test_schema_cache.py        - Schema 缓存单元测试，13 用例（无需 API Key）
  test_column_fuzzy_match.py  - 列名模糊匹配单元测试，20 用例（无需 API Key）
  test_chart_generation.py    - 图表生成单元测试，30 用例（无需 API Key）
  test_report_saving.py       - 报告保存单元测试，32 用例（无需 API Key）
  test_security.py            - SQL 安全护栏单元+集成测试，61 用例（无需 API Key）

运行方式:
  # 无需 API Key（156 个，全部通过）
  python tests/test_schema_cache.py
  python tests/test_column_fuzzy_match.py
  python tests/test_chart_generation.py
  python tests/test_report_saving.py
  python tests/test_security.py

  # 需要 API Key + DB
  python tests/test_router_accuracy.py
  python tests/test_analysis_detailed.py
"""
