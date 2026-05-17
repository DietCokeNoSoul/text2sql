<template>
  <div class="message-wrapper" :class="message.role">
    <!-- 用户消息 -->
    <div v-if="message.role === 'user'" class="bubble user-bubble">
      {{ message.content }}
    </div>

    <!-- AI 消息：步骤时间线 -->
    <div v-else class="ai-message">
      <!-- 有步骤数据（流式 / 刚完成） -->
      <template v-if="message.steps?.length">
        <template v-for="(item, i) in groupedSteps" :key="i">

          <!-- 普通路由步骤 -->
          <div v-if="item.type === 'plain'" class="step-row type-plain">
            <span class="step-label-plain">{{ item.label }}</span>
            <span v-if="item.status === 'done'" class="step-check-inline">✓</span>
            <span v-else class="step-dots-ani"><span/><span/><span/></span>
          </div>

          <!-- SQL 步骤组（流式期间展开，完成后二级折叠） -->
          <div v-else-if="item.type === 'sql-group'" class="step-row type-sql">
            <template v-if="message.streaming">
              <!-- 流式中：平铺显示 -->
              <div v-for="(s, si) in item.items" :key="si">
                <div class="step-header-row">
                  <span class="step-section-label">{{ s.label }}</span>
                  <span v-if="s.status === 'done'" class="step-check">✓</span>
                  <span v-else-if="s.status === 'skipped'" class="step-skip">跳过</span>
                  <span v-if="typeof s.elapsedMs === 'number'" class="step-elapsed">{{ formatElapsed(s.elapsedMs) }}</span>
                  <span v-else class="step-dots-ani"><span/><span/><span/></span>
                </div>
                <div v-if="s.content" class="sql-block">
                  <pre><code>{{ s.content }}</code></pre>
                </div>
                <div v-if="s.performance || s.optimization" class="perf-card">
                  <div class="perf-row">
                    <span class="perf-k">评分</span>
                    <span class="perf-v">{{ perfScoreText(s.performance, s.optimization) }}</span>
                  </div>
                  <div class="perf-row" v-if="s.performance">
                    <span class="perf-k">索引</span>
                    <span class="perf-v">{{ s.performance.uses_index ? '命中' : '未命中' }}</span>
                  </div>
                  <div class="perf-row" v-if="s.performance">
                    <span class="perf-k">全表扫描</span>
                    <span class="perf-v">{{ s.performance.full_scan ? '是' : '否' }}</span>
                  </div>
                  <div class="perf-row" v-if="s.optimization">
                    <span class="perf-k">自动替换</span>
                    <span class="perf-v">{{ s.optimization.optimized ? '已替换' : '未替换' }}</span>
                  </div>
                  <div class="perf-row" v-if="s.optimization && s.optimization.semantic_check_passed !== null && s.optimization.semantic_check_passed !== undefined">
                    <span class="perf-k">语义校验</span>
                    <span class="perf-v">{{ s.optimization.semantic_check_passed ? '通过' : '未通过' }}</span>
                  </div>
                </div>
              </div>
            </template>
            <template v-else>
              <!-- 完成后：一级折叠入口 -->
              <div class="step-header-row clickable" @click="toggleSqlGroup">
                <span class="step-section-label">SQL 查询</span>
                <span class="step-check">✓</span>
                <span v-if="sqlGroupTotalElapsed(item.items) !== null" class="step-elapsed">{{ formatElapsed(sqlGroupTotalElapsed(item.items)) }}</span>
                <span class="sql-toggle" :class="{ expanded: sqlGroupExpanded }">▾</span>
              </div>
              <!-- 一级展开内容 -->
              <div v-if="sqlGroupExpanded" class="sql-group-body">
                <!-- 只有一条 SQL：直接显示代码 -->
                <template v-if="item.items.length === 1">
                  <div class="step-header-row child single-sql-row">
                    <span class="step-section-label child-label">{{ item.items[0].label }}</span>
                    <span v-if="item.items[0].status === 'skipped'" class="step-skip">跳过</span>
                    <span v-else class="step-check">✓</span>
                    <span v-if="typeof item.items[0].elapsedMs === 'number'" class="step-elapsed">{{ formatElapsed(item.items[0].elapsedMs) }}</span>
                  </div>
                  <div v-if="item.items[0].content" class="sql-block">
                    <pre><code>{{ item.items[0].content }}</code></pre>
                  </div>
                  <div v-if="item.items[0].performance || item.items[0].optimization" class="perf-card">
                    <div class="perf-row">
                      <span class="perf-k">评分</span>
                      <span class="perf-v">{{ perfScoreText(item.items[0].performance, item.items[0].optimization) }}</span>
                    </div>
                    <div class="perf-row" v-if="item.items[0].performance">
                      <span class="perf-k">索引</span>
                      <span class="perf-v">{{ item.items[0].performance.uses_index ? '命中' : '未命中' }}</span>
                    </div>
                    <div class="perf-row" v-if="item.items[0].performance">
                      <span class="perf-k">全表扫描</span>
                      <span class="perf-v">{{ item.items[0].performance.full_scan ? '是' : '否' }}</span>
                    </div>
                    <div class="perf-row" v-if="item.items[0].optimization">
                      <span class="perf-k">自动替换</span>
                      <span class="perf-v">{{ item.items[0].optimization.optimized ? '已替换' : '未替换' }}</span>
                    </div>
                    <div class="perf-row" v-if="item.items[0].optimization && item.items[0].optimization.semantic_check_passed !== null && item.items[0].optimization.semantic_check_passed !== undefined">
                      <span class="perf-k">语义校验</span>
                      <span class="perf-v">{{ item.items[0].optimization.semantic_check_passed ? '通过' : '未通过' }}</span>
                    </div>
                  </div>
                  <div v-else-if="item.items[0].note" class="perf-card">
                    <div class="perf-row">
                      <span class="perf-k">备注</span>
                      <span class="perf-v perf-note">{{ item.items[0].note }}</span>
                    </div>
                  </div>
                </template>
                <!-- 多条 SQL：二级折叠列表 -->
                <template v-else>
                  <div v-for="(s, si) in item.items" :key="si" class="sql-child">
                    <div class="step-header-row clickable child" @click="toggleSql(si)">
                      <span class="step-section-label child-label">{{ s.label }}</span>
                      <span v-if="s.status === 'skipped'" class="step-skip">跳过</span>
                      <span v-else class="step-check">✓</span>
                      <span v-if="typeof s.elapsedMs === 'number'" class="step-elapsed">{{ formatElapsed(s.elapsedMs) }}</span>
                      <span v-if="s.content" class="sql-toggle" :class="{ expanded: expandedSqls.has(si) }">▾</span>
                    </div>
                    <div v-if="s.content && expandedSqls.has(si)" class="sql-block">
                      <pre><code>{{ s.content }}</code></pre>
                    </div>
                    <div v-if="expandedSqls.has(si) && (s.performance || s.optimization)" class="perf-card">
                      <div class="perf-row">
                        <span class="perf-k">评分</span>
                        <span class="perf-v">{{ perfScoreText(s.performance, s.optimization) }}</span>
                      </div>
                      <div class="perf-row" v-if="s.performance">
                        <span class="perf-k">索引</span>
                        <span class="perf-v">{{ s.performance.uses_index ? '命中' : '未命中' }}</span>
                      </div>
                      <div class="perf-row" v-if="s.performance">
                        <span class="perf-k">全表扫描</span>
                        <span class="perf-v">{{ s.performance.full_scan ? '是' : '否' }}</span>
                      </div>
                      <div class="perf-row" v-if="s.optimization">
                        <span class="perf-k">自动替换</span>
                        <span class="perf-v">{{ s.optimization.optimized ? '已替换' : '未替换' }}</span>
                      </div>
                      <div class="perf-row" v-if="s.optimization && s.optimization.semantic_check_passed !== null && s.optimization.semantic_check_passed !== undefined">
                        <span class="perf-k">语义校验</span>
                        <span class="perf-v">{{ s.optimization.semantic_check_passed ? '通过' : '未通过' }}</span>
                      </div>
                    </div>
                    <div v-else-if="expandedSqls.has(si) && s.note" class="perf-card">
                      <div class="perf-row">
                        <span class="perf-k">备注</span>
                        <span class="perf-v perf-note">{{ s.note }}</span>
                      </div>
                    </div>
                  </div>
                </template>
              </div>
            </template>
          </div>

          <!-- 回答步骤 -->
          <div v-else-if="item.type === 'answer'" class="step-row type-answer">
            <div class="step-header-row">
              <span class="step-section-label">{{ item.content ? '回答' : '正在处理…' }}</span>
              <span v-if="item.status === 'streaming'" class="cursor">▍</span>
            </div>
            <div v-if="item.content" class="answer-content markdown-body">
              <span v-html="renderMarkdown(item.content)" />
            </div>
            <div v-else-if="item.status !== 'done'" class="inline-wait">
              <span class="step-dots-ani"><span/><span/><span/></span>
            </div>
          </div>

        </template>
      </template>

      <!-- 初始等待（还没有任何步骤） -->
      <div v-else-if="message.streaming" class="init-wait">
        <span class="step-dots-ani"><span/><span/><span/></span>
      </div>

      <!-- 历史消息（只有 content，无 steps） -->
      <div v-else-if="message.content" class="bubble ai-bubble">
        <div v-html="renderMarkdown(message.content)" class="markdown-body" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  message: { type: Object, required: true },
})

// 一级：SQL 组是否展开
const sqlGroupExpanded = ref(false)
// 二级：各子 SQL 是否展开（Set of index）
const expandedSqls = ref(new Set())

// 将 steps 数组中的所有 sql 类型聚合为一个 sql-group
const groupedSteps = computed(() => {
  const steps = props.message.steps || []
  const result = []
  let sqlBuf = []

  const flushSql = () => {
    if (sqlBuf.length) {
      result.push({ type: 'sql-group', items: sqlBuf })
      sqlBuf = []
    }
  }

  for (const step of steps) {
    if (step.type === 'sql') {
      sqlBuf.push(step)
    } else {
      flushSql()
      result.push(step)
    }
  }
  flushSql()
  return result
})

function toggleSqlGroup() {
  sqlGroupExpanded.value = !sqlGroupExpanded.value
}

function toggleSql(i) {
  const s = new Set(expandedSqls.value)
  if (s.has(i)) s.delete(i)
  else s.add(i)
  expandedSqls.value = s
}

// 流式结束时，重置折叠状态（全部收起）
watch(() => props.message.streaming, (streaming) => {
  if (!streaming) {
    sqlGroupExpanded.value = false
    expandedSqls.value = new Set()
  }
})

function renderMarkdown(text) {
  return marked.parse(text ?? '')
}

function formatElapsed(ms) {
  return `${Math.max(0, Math.round(ms))}ms`
}

function sqlGroupTotalElapsed(items) {
  const values = (items || [])
    .map(item => item?.elapsedMs)
    .filter(value => typeof value === 'number')
  if (!values.length) return null
  return values.reduce((sum, value) => sum + value, 0)
}

function perfScoreText(perf, opt) {
  const after = perf?.score
  const before = opt?.original_analysis?.score
  if (typeof before === 'number' && typeof after === 'number') {
    return `${before} → ${after} / 100`
  }
  if (typeof after === 'number') {
    return `${after} / 100`
  }
  return '未知'
}
</script>

<style scoped>
.message-wrapper {
  display: flex;
  margin-bottom: 20px;
}
.message-wrapper.user  { justify-content: flex-end; }
.message-wrapper.assistant { justify-content: flex-start; }

/* 用户气泡 */
.bubble {
  max-width: 72%;
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
}
.user-bubble {
  background: #409eff;
  color: #fff;
  border-bottom-right-radius: 4px;
}

/* AI 消息容器 */
.ai-message {
  max-width: 86%;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

/* 历史消息 fallback */
.ai-bubble {
  background: #fff;
  border: 1px solid #e8eaed;
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}

/* ── 步骤行 ── */
.step-row {
  display: flex;
  flex-direction: column;
  padding: 0;
}

/* plain 步骤：单行（流式中暂时显示） */
.step-row.type-plain {
  flex-direction: row;
  align-items: center;
  gap: 6px;
  padding: 2px 0;
  color: #909399;
  font-size: 13px;
}
.step-label-plain { font-size: 13px; color: #909399; }
.step-check-inline { color: #67c23a; font-size: 12px; }

/* SQL / answer 步骤：块级 */
.step-row.type-sql  { padding: 4px 0 2px; }
.step-row.type-answer { padding: 8px 0 2px; }

/* 区块标题行（无圆点） */
.step-header-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

/* 区块标题文字 */
.step-section-label {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
}

.step-check { color: #67c23a; font-size: 13px; }
.step-skip  { color: #909399; font-size: 12px; }
.step-elapsed {
  color: #909399;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

/* 折叠箭头 */
.step-header-row.clickable {
  cursor: pointer;
  user-select: none;
}
.step-header-row.clickable:hover .step-section-label {
  color: #409eff;
}
.sql-toggle {
  margin-left: auto;
  font-size: 14px;
  color: #909399;
  transition: transform 0.2s ease;
  display: inline-block;
  transform: rotate(-90deg);
}
.sql-toggle.expanded {
  transform: rotate(0deg);
}

/* SQL 组内容区 */
.sql-group-body {
  margin-top: 2px;
}

.single-sql-row {
  margin-left: 12px;
}

/* 子级 SQL 项 */
.sql-child {
  margin-left: 12px;
  border-left: 2px solid #ebeef5;
  padding-left: 8px;
  margin-bottom: 4px;
}
.step-header-row.child {
  margin-bottom: 4px;
}
.child-label {
  font-size: 12px !important;
  font-weight: 500 !important;
  color: #606266 !important;
}

/* 等待动画（三点） */
.step-dots-ani {
  display: inline-flex;
  gap: 3px;
  align-items: center;
}
.step-dots-ani span {
  display: inline-block;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: #c0c4cc;
  animation: bounce 1.2s infinite;
}
.step-dots-ani span:nth-child(2) { animation-delay: .2s; }
.step-dots-ani span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce {
  0%,60%,100% { transform: translateY(0); }
  30%          { transform: translateY(-4px); }
}
.inline-wait { padding-left: 14px; }
.init-wait   { padding: 4px 0; }

/* SQL 代码块 */
.sql-block {
  margin-left: 14px;
  border-radius: 6px;
  overflow: hidden;
}
.sql-block pre {
  margin: 0;
  padding: 10px 14px;
  background: #1e1e2e;
  color: #cdd6f4;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-all;
}
.sql-block code { background: none; padding: 0; color: inherit; }

.perf-card {
  margin: 8px 0 0 14px;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid #e5eaf3;
  background: #f7f9fc;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.perf-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  line-height: 1.4;
}

.perf-k {
  color: #7a8599;
  min-width: 56px;
}

.perf-v {
  color: #2d3648;
  font-weight: 600;
}

.perf-note {
  font-weight: 500;
  white-space: normal;
  word-break: break-word;
}

/* 回答内容 */
.answer-content {
  margin-left: 14px;
  font-size: 14px;
  line-height: 1.7;
  color: #303133;
}

/* 光标 */
.cursor {
  animation: blink .8s step-end infinite;
  color: #409eff;
  font-weight: 700;
}
@keyframes blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0; }
}

/* Markdown */
.markdown-body :deep(p) { margin: 0 0 8px; }
.markdown-body :deep(p:last-child) { margin: 0; }
.markdown-body :deep(code) {
  background: #f0f2f5;
  padding: 1px 5px;
  border-radius: 3px;
  font-family: monospace;
  font-size: 13px;
}
.markdown-body :deep(pre) {
  background: #1e1e2e;
  color: #cdd6f4;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 6px 0;
}
.markdown-body :deep(pre code) { background: none; padding: 0; color: inherit; }
.markdown-body :deep(table) { border-collapse: collapse; width: 100%; margin: 8px 0; }
.markdown-body :deep(th), .markdown-body :deep(td) {
  border: 1px solid #e8eaed;
  padding: 6px 12px;
  text-align: left;
}
.markdown-body :deep(th) { background: #f5f7fa; font-weight: 600; }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 20px; margin: 4px 0; }
</style>
