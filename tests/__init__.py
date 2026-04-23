"""Text2SQL Agent 测试套件。

测试文件说明:
  test_import.py              - 核心符号导入冒烟测试，11 用例（无需 API Key）
  test_schema_cache.py        - Schema 缓存单元测试，13 用例（无需 API Key）
  test_column_fuzzy_match.py  - 列名模糊匹配单元测试，20 用例（无需 API Key）
  test_chart_generation.py    - 图表生成单元测试，30 用例（无需 API Key）
  test_report_saving.py       - 报告保存单元测试，32 用例（无需 API Key）
  test_security.py            - SQL 安全护栏单元+集成测试，61 用例（无需 API Key）
  test_retrieval_benchmark.py - 双塔检索架构基准测试，28 用例（无需 API Key + Milvus）
  test_session_plan.py        - SessionPlanManager 单元测试，21 用例（无需 API Key）
  test_placeholder_resolver.py - 占位符解析单元测试，9 用例（无需 API Key）
  test_dual_tower_node.py     - 双塔检索节点单元测试，4 用例（无需 API Key）
  test_cache_and_export.py    - LLM 缓存与导出单元测试，18 用例（无需 API Key）
  test_skill_registry.py      - SkillRegistry + _extract_skill_description 单元测试，15 用例（无需 API Key）
  test_session_plan_integration.py - SessionPlanManager 集成测试，3 用例（需 API Key）
  test_simple_skill.py        - Simple Query Skill 端到端测试（需 API Key）
  test_complex_detailed.py    - Complex Query Skill 详细测试（需 API Key）
  test_analysis_detailed.py   - Data Analysis Skill 详细测试（需 API Key）
  test_router_accuracy.py     - 主图路由准确率测试，15 个用例 100%（需 API Key）
  test_main_graph.py          - 主图集成测试（需 API Key）
  test_detailed_fix.py        - SQL 错误修复详细测试（需 API Key）

运行方式:
  # 无需 API Key（262 个，全部通过）
  pytest tests/ -k "not integration"
  # 或逐文件运行:
  python tests/test_import.py
  python tests/test_schema_cache.py
  python tests/test_column_fuzzy_match.py
  python tests/test_chart_generation.py
  python tests/test_report_saving.py
  python tests/test_security.py
  python tests/test_retrieval_benchmark.py
  python tests/test_session_plan.py
  python tests/test_placeholder_resolver.py
  python tests/test_dual_tower_node.py
  python tests/test_cache_and_export.py
  python tests/test_skill_registry.py

  # 需要 API Key + DB
  python tests/test_router_accuracy.py
  python tests/test_analysis_detailed.py
"""
