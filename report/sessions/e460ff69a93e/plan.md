# Find the top 3 shops by voucher count and show their average voucher values

**Task ID**: `e460ff69a93e`  
**Skill**: `complex_query`  
**Status**: done  
**Created**: 2026-04-23 16:56:10  
**Updated**: 2026-04-23 16:56:10

## 任务描述

Find the top 3 shops by voucher count and show their average voucher values

## 步骤进度

| 步骤 | 描述 | 状态 | 开始时间 | 完成时间 |
|------|------|------|----------|----------|
| Step 1 | Count vouchers per shop and get top 3 shops by count | [done] done | 2026-04-23 16:56:10 | 2026-04-23 16:56:10 |
| Step 2 | Calculate average voucher value for the top 3 shops | [done] done | 2026-04-23 16:56:10 | 2026-04-23 16:56:10 |

### Step 1 — Count vouchers per shop and get top 3 shops by count [done]

**SQL**:
```sql
SELECT shop_id, COUNT(*) AS voucher_count FROM tb_voucher GROUP BY shop_id ORDER BY voucher_count DESC LIMIT 3
```

**结果**: [(1, 2)]

### Step 2 — Calculate average voucher value for the top 3 shops [done]

**SQL**:
```sql
SELECT shop_id, AVG(actual_value) AS avg_actual_value FROM tb_voucher WHERE shop_id IN (1) GROUP BY shop_id
```

**结果**: [(1, Decimal('7500.0000'))]
