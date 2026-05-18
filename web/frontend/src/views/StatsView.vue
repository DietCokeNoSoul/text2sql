<template>
  <div class="stats-view">
    <!-- KPI 卡片行 -->
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-icon">🔢</div>
        <div class="kpi-body">
          <div class="kpi-value">{{ fmt(overview.total_queries) }}</div>
          <div class="kpi-label">累计查询次数</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">📅</div>
        <div class="kpi-body">
          <div class="kpi-value">{{ fmt(overview.month_queries) }}</div>
          <div class="kpi-label">本月查询次数</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">💬</div>
        <div class="kpi-body">
          <div class="kpi-value">{{ fmtK(overview.total_tokens) }}</div>
          <div class="kpi-label">累计 Token 消耗</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">📝</div>
        <div class="kpi-body">
          <div class="kpi-value">{{ fmt(overview.total_sql) }}</div>
          <div class="kpi-label">累计 SQL 生成条数</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">✅</div>
        <div class="kpi-body">
          <div class="kpi-value">{{ overview.success_rate?.toFixed(1) ?? '—' }}%</div>
          <div class="kpi-label">查询成功率</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-body">
          <div class="kpi-value">{{ fmtMs(overview.avg_latency_ms) }}</div>
          <div class="kpi-label">平均响应时长</div>
        </div>
      </div>
    </div>

    <!-- 图表区 -->
    <div class="charts-grid">
      <!-- 月度查询趋势 -->
      <div class="chart-card wide">
        <div class="chart-title">月度查询量趋势</div>
        <div ref="queryChartRef" class="chart-container" />
      </div>

      <!-- 月度 Token 趋势 -->
      <div class="chart-card wide">
        <div class="chart-title">月度 Token 消耗趋势</div>
        <div ref="tokenChartRef" class="chart-container" />
      </div>

      <!-- 技能分布饼图 -->
      <div class="chart-card narrow">
        <div class="chart-title">技能类型分布</div>
        <div ref="skillChartRef" class="chart-container" />
      </div>

      <!-- 平均延迟折线图 -->
      <div class="chart-card narrow">
        <div class="chart-title">月度平均响应时长（ms）</div>
        <div ref="latencyChartRef" class="chart-container" />
      </div>
    </div>

    <!-- 空数据提示 -->
    <div v-if="!loading && monthly.length === 0 && overview.total_queries === 0" class="empty-tip">
      <span>📊 暂无统计数据，发起查询后将自动记录</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { getStatsOverview, getMonthlyStats } from '../api/stats.js'

const overview = ref({})
const monthly = ref([])
const loading = ref(true)

const queryChartRef  = ref(null)
const tokenChartRef  = ref(null)
const skillChartRef  = ref(null)
const latencyChartRef = ref(null)

let queryChart   = null
let tokenChart   = null
let skillChart   = null
let latencyChart = null

// ── 格式化 ────────────────────────────────────────────────────────────────────
function fmt(n)   { return n != null ? Number(n).toLocaleString() : '—' }
function fmtK(n)  {
  if (n == null) return '—'
  return n >= 1000 ? (n / 1000).toFixed(1) + 'K' : String(Number(n))
}
function fmtMs(n) {
  if (n == null || n === 0) return '—'
  return n >= 1000 ? (n / 1000).toFixed(1) + 's' : Math.round(n) + 'ms'
}

// ── ECharts 通用颜色 ─────────────────────────────────────────────────────────
const COLOR_PRIMARY   = '#409EFF'
const COLOR_SUCCESS   = '#67C23A'
const COLOR_WARNING   = '#E6A23C'
const COLOR_DANGER    = '#F56C6C'
const COLOR_INFO      = '#909399'

const SKILL_LABEL = {
  simple_query:   '简单查询',
  complex_query:  '复杂查询',
  data_analysis:  '数据分析',
  general_chat:   '通用对话',
  unknown:        '未知',
}

// ── 渲染图表 ──────────────────────────────────────────────────────────────────
function renderCharts() {
  const echarts = window.echarts
  if (!echarts) return

  const months  = monthly.value.map(r => r.month)
  const queries = monthly.value.map(r => r.query_count)
  const sqlCnts = monthly.value.map(r => r.sql_count)
  const prompts = monthly.value.map(r => r.prompt_tokens)
  const compls  = monthly.value.map(r => r.completion_tokens)
  const latency = monthly.value.map(r => r.avg_latency_ms)

  // 1. 月度查询量（双Y轴：查询次数 + SQL条数）
  if (queryChartRef.value) {
    queryChart = queryChart || echarts.init(queryChartRef.value)
    queryChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['查询次数', 'SQL 生成条数'], bottom: 0 },
      grid: { left: '3%', right: '4%', bottom: '12%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: months },
      yAxis: [
        { type: 'value', name: '查询次数' },
        { type: 'value', name: 'SQL 条数' },
      ],
      series: [
        {
          name: '查询次数',
          type: 'line',
          smooth: true,
          data: queries,
          itemStyle: { color: COLOR_PRIMARY },
          areaStyle: { color: 'rgba(64,158,255,0.15)' },
        },
        {
          name: 'SQL 生成条数',
          type: 'line',
          smooth: true,
          yAxisIndex: 1,
          data: sqlCnts,
          itemStyle: { color: COLOR_SUCCESS },
          areaStyle: { color: 'rgba(103,194,58,0.12)' },
        },
      ],
    })
  }

  // 2. 月度 Token 趋势（堆叠面积）
  if (tokenChartRef.value) {
    tokenChart = tokenChart || echarts.init(tokenChartRef.value)
    tokenChart.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      legend: { data: ['Prompt Tokens', 'Completion Tokens'], bottom: 0 },
      grid: { left: '3%', right: '4%', bottom: '12%', containLabel: true },
      xAxis: { type: 'category', boundaryGap: false, data: months },
      yAxis: { type: 'value', name: 'Tokens' },
      series: [
        {
          name: 'Prompt Tokens',
          type: 'line',
          stack: 'tokens',
          smooth: true,
          data: prompts,
          itemStyle: { color: COLOR_WARNING },
          areaStyle: { color: 'rgba(230,162,60,0.2)' },
        },
        {
          name: 'Completion Tokens',
          type: 'line',
          stack: 'tokens',
          smooth: true,
          data: compls,
          itemStyle: { color: COLOR_DANGER },
          areaStyle: { color: 'rgba(245,108,108,0.2)' },
        },
      ],
    })
  }

  // 3. 技能分布饼图
  if (skillChartRef.value) {
    skillChart = skillChart || echarts.init(skillChartRef.value)
    const dist = (overview.value.skill_distribution || []).map(d => ({
      name:  SKILL_LABEL[d.skill_type] || d.skill_type,
      value: d.count,
    }))
    skillChart.setOption({
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { orient: 'vertical', right: 10, top: 'center' },
      series: [
        {
          name: '技能类型',
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: false,
          label: { show: false },
          emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } },
          data: dist.length ? dist : [{ name: '暂无数据', value: 1, itemStyle: { color: '#e0e0e0' } }],
        },
      ],
    })
  }

  // 4. 平均延迟折线图
  if (latencyChartRef.value) {
    latencyChart = latencyChart || echarts.init(latencyChartRef.value)
    latencyChart.setOption({
      tooltip: { trigger: 'axis', formatter: params => `${params[0].axisValue}<br/>${params[0].marker}平均延迟: ${Math.round(params[0].value)} ms` },
      grid: { left: '3%', right: '4%', bottom: '8%', containLabel: true },
      xAxis: { type: 'category', data: months },
      yAxis: { type: 'value', name: 'ms' },
      series: [
        {
          name: '平均延迟',
          type: 'line',
          smooth: true,
          data: latency,
          itemStyle: { color: COLOR_INFO },
          lineStyle: { width: 2 },
          symbol: 'circle',
          symbolSize: 6,
        },
      ],
    })
  }
}

function resizeCharts() {
  queryChart?.resize()
  tokenChart?.resize()
  skillChart?.resize()
  latencyChart?.resize()
}

// ── 数据加载 ──────────────────────────────────────────────────────────────────
async function loadData() {
  loading.value = true
  try {
    const [ov, ms] = await Promise.all([getStatsOverview(), getMonthlyStats(12)])
    overview.value = ov
    monthly.value = ms.monthly || []
    await nextTick()
    renderCharts()
  } catch (e) {
    console.error('Failed to load stats:', e)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadData()
  window.addEventListener('resize', resizeCharts)
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeCharts)
  queryChart?.dispose()
  tokenChart?.dispose()
  skillChart?.dispose()
  latencyChart?.dispose()
})
</script>

<style scoped>
.stats-view {
  height: 100%;
  overflow-y: auto;
  padding: 20px 24px;
  background: #f5f7fa;
  box-sizing: border-box;
}

/* KPI 卡片 */
.kpi-row {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.kpi-card {
  flex: 1;
  min-width: 140px;
  background: #fff;
  border-radius: 10px;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: 0 1px 6px rgba(0,0,0,.06);
}

.kpi-icon {
  font-size: 28px;
  flex-shrink: 0;
}

.kpi-value {
  font-size: 22px;
  font-weight: 700;
  color: #303133;
  line-height: 1.2;
}

.kpi-label {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

/* 图表区 */
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.chart-card {
  background: #fff;
  border-radius: 10px;
  padding: 16px;
  box-shadow: 0 1px 6px rgba(0,0,0,.06);
}

.chart-card.wide {
  grid-column: span 2;
}

.chart-card.narrow {
  grid-column: span 1;
}

.chart-title {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 12px;
}

.chart-container {
  width: 100%;
  height: 260px;
}

/* 空数据提示 */
.empty-tip {
  text-align: center;
  padding: 40px;
  color: #909399;
  font-size: 14px;
}

@media (max-width: 900px) {
  .charts-grid { grid-template-columns: 1fr; }
  .chart-card.wide,
  .chart-card.narrow { grid-column: span 1; }
}
</style>
