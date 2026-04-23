# 统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺名称和优惠券数量

**Task ID**: `607d613c7362`  
**Skill**: `complex_query`  
**Status**: done  
**Created**: 2026-04-23 17:23:06  
**Updated**: 2026-04-23 17:23:06

## 任务描述

统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺名称和优惠券数量

## 步骤进度

| 步骤 | 描述 | 状态 | 开始时间 | 完成时间 |
|------|------|------|----------|----------|
| Step 1 | 统计每家店铺的优惠券数量，并按数量降序排列，取前5名 | [done] done | 2026-04-23 17:23:06 | 2026-04-23 17:23:06 |
| Step 2 | 获取前5名店铺的名称和优惠券数量，通过shop_id关联tb_shop表 | [done] done | 2026-04-23 17:23:06 | 2026-04-23 17:23:06 |

### Step 1 — 统计每家店铺的优惠券数量，并按数量降序排列，取前5名 [done]

**SQL**:
```sql
SELECT shop_id, COUNT(*) AS voucher_count FROM tb_voucher GROUP BY shop_id ORDER BY voucher_count DESC LIMIT 5
```

**结果**: [(1, 2)]

### Step 2 — 获取前5名店铺的名称和优惠券数量，通过shop_id关联tb_shop表 [done]

**SQL**:
```sql
SELECT s.name, v.voucher_count FROM tb_shop s INNER JOIN (SELECT shop_id, COUNT(*) AS voucher_count FROM tb_voucher GROUP BY shop_id ORDER BY voucher_count DESC LIMIT 5) v ON s.id = v.shop_id
```

**结果**: [('101茶餐厅', 2)]
