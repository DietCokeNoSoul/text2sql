# Chinook 数据库中哪些艺术家最受欢迎？分析他们的专辑数、总曲目数和平均曲目时长。

**Task ID**: `18d968c38edf`  
**Skill**: `complex_query`  
**Status**: done  
**Created**: 2026-04-21 22:35:30  
**Updated**: 2026-04-21 22:35:30

## 任务描述

Chinook 数据库中哪些艺术家最受欢迎？分析他们的专辑数、总曲目数和平均曲目时长。

## 步骤进度

| 步骤 | 描述 | 状态 | 开始时间 | 完成时间 |
|------|------|------|----------|----------|
| Step 1 | 查询所有艺术家及其专辑数量 | [done] done | 2026-04-21 22:35:30 | 2026-04-21 22:35:30 |
| Step 2 | 统计专辑数量最多的艺术家的总曲目数 | [done] done | 2026-04-21 22:35:30 | 2026-04-21 22:35:30 |
| Step 3 | 查找这些热门艺术家的平均曲目时长（毫秒转分钟） | [done] done | 2026-04-21 22:35:30 | 2026-04-21 22:35:30 |

### Step 1 — 查询所有艺术家及其专辑数量 [done]

**SQL**:
```sql
SELECT ar.Name AS 艺术家, COUNT(al.AlbumId) AS 专辑数 FROM Artist ar LEFT JOIN Album al ON ar.ArtistId = al.ArtistId GROUP BY ar.ArtistId ORDER BY 专辑数 DESC LIMIT 10
```

**结果**: [('Iron Maiden', 21), ('Led Zeppelin', 14), ('Deep Purple', 11), ('Metallica', 10), ('U2', 10), ('Ozzy Osbourne', 6), ('Pearl Jam', 5), ('Various Artists', 4), ('Faith No More', 4), ('Foo Fighters', 4

### Step 2 — 统计专辑数量最多的艺术家的总曲目数 [done]

**SQL**:
```sql
SELECT ar.Name AS 艺术家, COUNT(t.TrackId) AS 总曲目数 FROM Artist ar JOIN Album al ON ar.ArtistId = al.ArtistId JOIN Track t ON al.AlbumId = t.AlbumId GROUP BY ar.ArtistId ORDER BY 总曲目数 DESC LIMIT 5
```

**结果**: [('Iron Maiden', 213), ('U2', 135), ('Led Zeppelin', 114), ('Metallica', 112), ('Lost', 92)]

### Step 3 — 查找这些热门艺术家的平均曲目时长（毫秒转分钟） [done]

**SQL**:
```sql
SELECT ar.Name AS 艺术家, ROUND(AVG(t.Milliseconds) / 60000.0, 2) AS 平均时长_分钟 FROM Artist ar JOIN Album al ON ar.ArtistId = al.ArtistId JOIN Track t ON al.AlbumId = t.AlbumId GROUP BY ar.ArtistId ORDER BY 平均时长_分钟 DESC LIMIT 5
```

**结果**: [('Battlestar Galactica (Classic)', 48.76), ('Battlestar Galactica', 46.17), ('Heroes', 43.32), ('Lost', 43.17), ('Aquaman', 41.41)]
