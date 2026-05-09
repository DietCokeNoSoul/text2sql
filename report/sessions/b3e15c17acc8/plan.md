# 统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺名称和优惠券数量

**Task ID**: `b3e15c17acc8`  
**Skill**: `complex_query`  
**Status**: done  
**Created**: 2026-05-09 15:51:39  
**Updated**: 2026-05-09 15:51:39

## 任务描述

统计每家店铺拥有的优惠券数量，并按数量从高到低排列，显示前5名店铺名称和优惠券数量

## 步骤进度

| 步骤 | 描述 | 状态 | 开始时间 | 完成时间 |
|------|------|------|----------|----------|
| Step 1 | 统计每家店铺的优惠券数量，并按数量降序排列，取前5名 | [done] done | 2026-05-09 15:51:39 | 2026-05-09 15:51:39 |

### Step 1 — 统计每家店铺的优惠券数量，并按数量降序排列，取前5名 [done]

**SQL**:
```sql
SELECT s.name, COUNT(v.id) AS voucher_count FROM tb_shop s LEFT JOIN tb_voucher v ON s.id = v.shop_id GROUP BY s.id, s.name ORDER BY voucher_count DESC LIMIT 5
```

**结果**: [('101茶餐厅', 2), ('蔡馬洪涛烤肉·老北京铜锅涮羊肉', 0), ('新白鹿餐厅(运河上街店)', 0), ('Mamala(杭州远洋乐堤港店)', 0), ('海底捞火锅(水晶城购物中心店）', 0)]
