# 双塔检索基准测试报告

> 生成时间：2026-04-23 15:37:45  
> 数据库：Chinook.db  
> Schema 表数：11  |  外键关系：11

## 测试背景

对比有 / 无双塔检索（Milvus 列向量 + Steiner Tree 路径规划）时，
LLM 接收的 Schema 大小和 token 消耗差异。

- **Before**：全量 Schema 注入 LLM prompt（无任何剪枝）
- **After** ：双塔检索剪枝后的 Schema（仅含相关表及 JOIN 路径）

## 全局指标

| 指标 | 值 |
|------|-----|
| 全量 Schema | 5,881 字符 / ~1,680 tokens |
| 平均剪枝率 | **65.2%** |
| 平均每条查询节省 | **~1095 tokens** |
| 10 条查询总节省 | ~10956 tokens |
| 表命中准确率 | **100.0%** |
| 平均检索延迟 | 1.9ms (SchemaGraph, 不含 Milvus) |

## 逐条查询对比

| ID | 查询 | 复杂度 | Before(tokens) | After(tokens) | 节省% | 准确率 | 延迟(ms) |
|----|------|--------|---------------|--------------|-------|--------|---------|
| Q01 | 查询每位摇滚艺术家的专辑数量和总曲目数 | 高 | 1,680 | 448 | 73% | 100% | 3 |
| Q02 | 找出消费金额最高的前5名客户及其购买的发票总数 | 中 | 1,680 | 528 | 69% | 100% | 1 |
| Q03 | 统计每位员工管理的客户数量及其产生的总销售额 | 高 | 1,680 | 871 | 48% | 100% | 2 |
| Q04 | 查询包含超过20首歌的播放列表及所有歌曲的总时长 | 中 | 1,680 | 398 | 76% | 100% | 2 |
| Q05 | 找出购买过Jazz类型歌曲的所有客户姓名和邮箱 | 极高 | 1,680 | 962 | 43% | 100% | 2 |
| Q06 | 统计每种媒体格式下歌曲的平均时长和平均文件大小 | 低 | 1,680 | 316 | 81% | 100% | 2 |
| Q07 | 查询每位艺术家最畅销的专辑（按发票行数统计） | 高 | 1,680 | 535 | 68% | 100% | 2 |
| Q08 | 找出同一员工服务的来自同一国家的客户，以及这些客户的总消费 | 高 | 1,680 | 871 | 48% | 100% | 3 |
| Q09 | 统计2009年每个月的销售总额、订单数和平均每单金额 | 中 | 1,680 | 361 | 78% | 100% | 1 |
| Q10 | 查询包含特定艺术家歌曲的所有播放列表名称及歌曲数量 | 极高 | 1,680 | 551 | 67% | 100% | 3 |

## 详细步骤

### Q01：查询每位摇滚艺术家的专辑数量和总曲目数

- **复杂度**：高
- **所需表**：Artist, Album, Track, Genre
- **Steiner Tree 路径**：Track → Artist → Genre → Album
- **Before**：1,680 tokens
- **After** ：448 tokens（节省 73%，~1232 tokens）
- **表命中准确率**：100%
- **检索延迟**：3.1ms

### Q02：找出消费金额最高的前5名客户及其购买的发票总数

- **复杂度**：中
- **所需表**：Customer, Invoice
- **Steiner Tree 路径**：Invoice → Customer
- **Before**：1,680 tokens
- **After** ：528 tokens（节省 69%，~1151 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.2ms

### Q03：统计每位员工管理的客户数量及其产生的总销售额

- **复杂度**：高
- **所需表**：Employee, Customer, Invoice
- **Steiner Tree 路径**：Invoice → Employee → Customer
- **Before**：1,680 tokens
- **After** ：871 tokens（节省 48%，~808 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.5ms

### Q04：查询包含超过20首歌的播放列表及所有歌曲的总时长

- **复杂度**：中
- **所需表**：Playlist, PlaylistTrack, Track
- **Steiner Tree 路径**：Track → Playlist → PlaylistTrack
- **Before**：1,680 tokens
- **After** ：398 tokens（节省 76%，~1281 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.5ms

### Q05：找出购买过Jazz类型歌曲的所有客户姓名和邮箱

- **复杂度**：极高
- **所需表**：Customer, Invoice, InvoiceLine, Track, Genre
- **Steiner Tree 路径**：InvoiceLine → Track → Invoice → Genre → Customer
- **Before**：1,680 tokens
- **After** ：962 tokens（节省 43%，~718 tokens）
- **表命中准确率**：100%
- **检索延迟**：2.0ms

### Q06：统计每种媒体格式下歌曲的平均时长和平均文件大小

- **复杂度**：低
- **所需表**：Track, MediaType
- **Steiner Tree 路径**：MediaType → Track
- **Before**：1,680 tokens
- **After** ：316 tokens（节省 81%，~1363 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.5ms

### Q07：查询每位艺术家最畅销的专辑（按发票行数统计）

- **复杂度**：高
- **所需表**：Artist, Album, Track, InvoiceLine
- **Steiner Tree 路径**：InvoiceLine → Track → Artist → Album
- **Before**：1,680 tokens
- **After** ：535 tokens（节省 68%，~1144 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.5ms

### Q08：找出同一员工服务的来自同一国家的客户，以及这些客户的总消费

- **复杂度**：高
- **所需表**：Employee, Customer, Invoice
- **Steiner Tree 路径**：Invoice → Employee → Customer
- **Before**：1,680 tokens
- **After** ：871 tokens（节省 48%，~808 tokens）
- **表命中准确率**：100%
- **检索延迟**：3.2ms

### Q09：统计2009年每个月的销售总额、订单数和平均每单金额

- **复杂度**：中
- **所需表**：Invoice, InvoiceLine
- **Steiner Tree 路径**：InvoiceLine → Invoice
- **Before**：1,680 tokens
- **After** ：361 tokens（节省 78%，~1318 tokens）
- **表命中准确率**：100%
- **检索延迟**：1.0ms

### Q10：查询包含特定艺术家歌曲的所有播放列表名称及歌曲数量

- **复杂度**：极高
- **所需表**：Playlist, PlaylistTrack, Track, Album, Artist
- **Steiner Tree 路径**：Artist → PlaylistTrack → Playlist → Track → Album
- **Before**：1,680 tokens
- **After** ：551 tokens（节省 67%，~1129 tokens）
- **表命中准确率**：100%
- **检索延迟**：2.6ms

## 结论

双塔检索架构（Milvus 列向量 + Steiner Tree 路径规划）在 Chinook.db 上
平均将 LLM 接收的 Schema 大小压缩 **65%**，
每条复杂查询平均节省 **~1095 tokens**。
在全部 10 条测试查询中，表命中准确率达 **100%**。

> 注：以上延迟仅含 SchemaGraph Steiner Tree 计算，实际部署时还需加上
> Milvus 向量检索耗时（通常 30~80ms）。