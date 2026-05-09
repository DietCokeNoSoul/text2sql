# 统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺名称和优惠券数量

**Task ID**: `243939da3ec7`  
**Skill**: `complex_query`  
**Status**: done  
**Created**: 2026-05-09 15:51:05  
**Updated**: 2026-05-09 15:51:05

## 任务描述

统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺名称和优惠券数量

## 步骤进度

| 步骤 | 描述 | 状态 | 开始时间 | 完成时间 |
|------|------|------|----------|----------|
| Step 1 | 统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺ID和优惠券数量 | [done] done | 2026-05-09 15:51:05 | 2026-05-09 15:51:05 |
| Step 2 | 获取前5名店铺的名称和优惠券数量，通过关联tb_shop表 | [done] done | 2026-05-09 15:51:05 | 2026-05-09 15:51:05 |

### Step 1 — 统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺ID和优惠券数量 [done]

**SQL**:
```sql
SELECT shop_id, COUNT(*) AS voucher_count FROM tb_voucher GROUP BY shop_id ORDER BY voucher_count DESC LIMIT 5
```

**结果**: [(1, 2)]

### Step 2 — 获取前5名店铺的名称和优惠券数量，通过关联tb_shop表 [done]

**SQL**:
```sql
SELECT s.name, v.voucher_count FROM tb_shop s JOIN (SELECT shop_id, COUNT(*) AS voucher_count FROM tb_voucher GROUP BY shop_id ORDER BY voucher_count DESC LIMIT 5) v ON s.id = v.shop_id
```

**结果**: [('101茶餐厅', 2)]
