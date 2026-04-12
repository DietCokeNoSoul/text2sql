"""Text2SQL Agent 测试套件。

测试文件说明:
  test_simple_skill.py      - Simple Query Skill 端到端测试
  test_complex_detailed.py  - Complex Query Skill 详细测试
  test_analysis_detailed.py - Data Analysis Skill 详细测试
  test_router_accuracy.py   - 主图路由准确率测试（15 个用例，100%）
  test_schema_cache.py      - Schema 缓存单元测试（无 DB 依赖）
  test_main_graph.py        - 主图集成测试
  test_detailed_fix.py      - SQL 错误修复详细测试

运行方式:
  python tests/test_schema_cache.py         # 无需 DB，纯单元测试
  python tests/test_router_accuracy.py      # 需要 LLM API Key
  python tests/test_analysis_detailed.py    # 需要 DB + LLM API Key
"""
