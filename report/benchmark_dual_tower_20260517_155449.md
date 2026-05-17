# 双塔检索基准测试报告

> 生成时间：2026-05-17 15:54:49  
> 数据库：Chinook.db  
> Schema 表数：2  |  外键关系：0

## 测试背景

对比有 / 无双塔检索（Milvus 列向量 + Steiner Tree 路径规划）时，
LLM 接收的 Schema 大小和 token 消耗差异。

- **Before**：全量 Schema 注入 LLM prompt（无任何剪枝）
- **After** ：双塔检索剪枝后的 Schema（仅含相关表及 JOIN 路径）

## 全局指标

| 指标 | 值 |
|------|-----|
| 全量 Schema | 284 字符 / ~81 tokens |
| 平均剪枝率 | **50.4%** |
| 平均每条查询节省 | **~40 tokens** |
| 10 条查询总节省 | ~408 tokens |
| 表命中准确率 | **100.0%** |
| 平均检索延迟 | 0.3ms (SchemaGraph, 不含 Milvus) |

## 逐条查询对比

| ID | 查询 | 复杂度 | Before(tokens) | After(tokens) | 节省% | 准确率 | 延迟(ms) |
|----|------|--------|---------------|--------------|-------|--------|---------|
| Q01 | 查询店铺列表 | 低 | 81 | 39 | 51% | 100% | 0 |
| Q02 | 查询用户列表 | 低 | 81 | 41 | 49% | 100% | 0 |
| Q03 | 按店铺名称排序 | 低 | 81 | 39 | 51% | 100% | 1 |
| Q04 | 按用户名排序 | 低 | 81 | 41 | 49% | 100% | 1 |
| Q05 | 查询店铺数量 | 低 | 81 | 39 | 51% | 100% | 0 |
| Q06 | 查询用户数量 | 低 | 81 | 41 | 49% | 100% | 0 |
| Q07 | 查询店铺 ID 和名称 | 低 | 81 | 39 | 51% | 100% | 0 |
| Q08 | 查询用户 ID 和用户名 | 低 | 81 | 41 | 49% | 100% | 1 |
| Q09 | 模糊查询店铺名称 | 低 | 81 | 39 | 51% | 100% | 0 |
| Q10 | 模糊查询用户名 | 低 | 81 | 41 | 49% | 100% | 0 |

## 详细步骤

### Q01：查询店铺列表

- **复杂度**：低
- **所需表**：tb_shop
- **Steiner Tree 路径**：tb_shop
- **Before**：81 tokens
- **After** ：39 tokens（节省 51%，~41 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.0ms

### Q02：查询用户列表

- **复杂度**：低
- **所需表**：tb_user
- **Steiner Tree 路径**：tb_user
- **Before**：81 tokens
- **After** ：41 tokens（节省 49%，~40 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.0ms

### Q03：按店铺名称排序

- **复杂度**：低
- **所需表**：tb_shop
- **Steiner Tree 路径**：tb_shop
- **Before**：81 tokens
- **After** ：39 tokens（节省 51%，~41 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.0ms

### Q04：按用户名排序

- **复杂度**：低
- **所需表**：tb_user
- **Steiner Tree 路径**：tb_user
- **Before**：81 tokens
- **After** ：41 tokens（节省 49%，~40 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.5ms

### Q05：查询店铺数量

- **复杂度**：低
- **所需表**：tb_shop
- **Steiner Tree 路径**：tb_shop
- **Before**：81 tokens
- **After** ：39 tokens（节省 51%，~41 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.0ms

### Q06：查询用户数量

- **复杂度**：低
- **所需表**：tb_user
- **Steiner Tree 路径**：tb_user
- **Before**：81 tokens
- **After** ：41 tokens（节省 49%，~40 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.0ms

### Q07：查询店铺 ID 和名称

- **复杂度**：低
- **所需表**：tb_shop
- **Steiner Tree 路径**：tb_shop
- **Before**：81 tokens
- **After** ：39 tokens（节省 51%，~41 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.0ms

### Q08：查询用户 ID 和用户名

- **复杂度**：低
- **所需表**：tb_user
- **Steiner Tree 路径**：tb_user
- **Before**：81 tokens
- **After** ：41 tokens（节省 49%，~40 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.0ms

### Q09：模糊查询店铺名称

- **复杂度**：低
- **所需表**：tb_shop
- **Steiner Tree 路径**：tb_shop
- **Before**：81 tokens
- **After** ：39 tokens（节省 51%，~41 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.0ms

### Q10：模糊查询用户名

- **复杂度**：低
- **所需表**：tb_user
- **Steiner Tree 路径**：tb_user
- **Before**：81 tokens
- **After** ：41 tokens（节省 49%，~40 tokens）
- **表命中准确率**：100%
- **检索延迟**：0.0ms

## 结论

双塔检索架构（Milvus 列向量 + Steiner Tree 路径规划）在 Chinook.db 上
平均将 LLM 接收的 Schema 大小压缩 **50%**，
每条复杂查询平均节省 **~40 tokens**。
在全部 10 条测试查询中，表命中准确率达 **100%**。

> 注：以上延迟仅含 SchemaGraph Steiner Tree 计算，实际部署时还需加上
> Milvus 向量检索耗时（通常 30~80ms）。