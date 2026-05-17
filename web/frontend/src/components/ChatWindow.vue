<template>
  <div class="chat-window">
    <!-- 消息列表 -->
    <div class="messages-container" ref="messagesContainer">
      <div v-if="messages.length === 0" class="empty-state">
        <el-empty description="开始提问，探索您的数据库" :image-size="80">
          <template #image>
            <span style="font-size: 60px">🤖</span>
          </template>
        </el-empty>
        <div class="suggestions">
          <el-tag
            v-for="s in suggestions"
            :key="s"
            class="suggestion-tag"
            @click="submitQuery(s)"
          >{{ s }}</el-tag>
        </div>
      </div>

      <MessageBubble
        v-for="(msg, i) in messages"
        :key="i"
        :message="msg"
      />
    </div>

    <!-- 输入区域 -->
    <div class="input-area">
      <el-input
        v-model="inputText"
        type="textarea"
        :rows="2"
        placeholder="输入您的问题，按 Enter 发送，Shift+Enter 换行"
        :disabled="isStreaming"
        @keydown.enter.exact.prevent="submitQuery()"
        resize="none"
        class="query-input"
      />
      <el-button
        type="primary"
        :loading="isStreaming"
        :disabled="!inputText.trim()"
        @click="submitQuery()"
        class="send-btn"
        size="large"
      >
        {{ isStreaming ? '生成中…' : '发送' }}
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch, onMounted, inject } from 'vue'
import MessageBubble from './MessageBubble.vue'
import { streamQuery, getHistory, getPlans } from '../api/chat.js'

const props = defineProps({
  threadId: { type: String, default: '' },
  sessionId: { type: String, default: '' },
})

const emit = defineEmits(['session-named', 'query-start', 'query-done'])

const openSqlConfirm = inject('openSqlConfirm')
const ensureSession = inject('ensureSession')

const messages = ref([])
const inputText = ref('')
const isStreaming = ref(false)
const messagesContainer = ref(null)

const suggestions = [
  '数据库中有多少条记录？',
  '查看所有表名',
  '最近10条数据',
]

function parseSqlDisplayFromNote(note) {
  if (!note || typeof note !== 'string') {
    return { elapsedMs: null, performance: null, optimization: null }
  }

  const elapsedMatch = note.match(/耗时\s*[:：]\s*(\d+)\s*ms/i)
  const elapsedMs = elapsedMatch ? Number(elapsedMatch[1]) : null

  let beforeScore = null
  let afterScore = null

  const scoreArrowMatch = note.match(/(?:评分|优化)\s*[:：]\s*(\d+)\s*(?:->|→)\s*(\d+)/)
  if (scoreArrowMatch) {
    beforeScore = Number(scoreArrowMatch[1])
    afterScore = Number(scoreArrowMatch[2])
  }

  const scoreSingleMatch = note.match(/性能评分\s*=\s*(\d+)\s*\/\s*100/)
  if (!scoreArrowMatch && scoreSingleMatch) {
    beforeScore = Number(scoreSingleMatch[1])
    afterScore = Number(scoreSingleMatch[1])
  }

  const hasIndexHit = note.includes('命中索引') && !note.includes('未命中索引')
  const hasIndexMiss = note.includes('未命中索引')
  const hasNoFullScan = note.includes('无明显全表扫描')
  const hasFullScan = !hasNoFullScan && note.includes('全表扫描')

  const performance =
    typeof afterScore === 'number'
      ? {
          score: afterScore,
          uses_index: hasIndexHit ? true : hasIndexMiss ? false : false,
          full_scan: hasNoFullScan ? false : hasFullScan ? true : false,
        }
      : null

  const optimization =
    typeof beforeScore === 'number' && typeof afterScore === 'number'
      ? {
          optimized: beforeScore !== afterScore,
          original_analysis: { score: beforeScore },
        }
      : null

  return { elapsedMs, performance, optimization }
}

// 切换 session 时重新加载历史
watch(() => props.threadId, async (newId) => {
  messages.value = []
  if (newId) {
    await loadHistory(newId)
  }
})

async function loadHistory(threadId) {
  const [{ messages: hist }, { plans }] = await Promise.all([
    getHistory(threadId),
    getPlans(threadId),
  ])

  // SQL 展示元信息（尤其耗时）统一以 plan 为准；同一 SQL 允许出现多次，按时间顺序消费。
  const sqlPlanMetaMap = new Map()
  ;(plans || [])
    .slice()
    .sort((a, b) => String(a?.created_at || '').localeCompare(String(b?.created_at || '')))
    .forEach(plan => {
      ;(plan?.steps || []).forEach(step => {
        const sql = step?.sql || ''
        if (!sql) return
        const arr = sqlPlanMetaMap.get(sql) || []
        arr.push({
          note: step?.notes || '',
          elapsedMs: typeof step?.elapsed_ms === 'number' ? step.elapsed_ms : null,
          stepId: String(step?.step_id ?? ''),
        })
        sqlPlanMetaMap.set(sql, arr)
      })
    })

  const sqlPlanMetaCursor = new Map()
  const consumePlanMeta = (sql) => {
    if (!sql || !sqlPlanMetaMap.has(sql)) return null
    const arr = sqlPlanMetaMap.get(sql) || []
    const idx = sqlPlanMetaCursor.get(sql) || 0
    const item = arr[idx] || arr[arr.length - 1] || null
    sqlPlanMetaCursor.set(sql, idx + 1)
    return item
  }

  messages.value = hist.map(m => {
    if (m.role !== 'assistant') return { role: m.role, content: m.content, steps: [], streaming: false }

    // 历史 AI 消息重建为 steps 格式，与流式完成后保持一致
    const steps = []
    const sqlSteps = Array.isArray(m.sql_steps) ? m.sql_steps : []
    if (sqlSteps.length) {
      sqlSteps.forEach((step, idx) => {
        const label = step?.label || (sqlSteps.length > 1 ? `SQL 查询 ${idx + 1}` : 'SQL 查询')
        const planMeta = consumePlanMeta(step?.sql || '') || {}
        const fallbackNote = planMeta.note || ''
        const parsedFromNote = parseSqlDisplayFromNote(fallbackNote)
        steps.push({
          key: `sql_${idx}`,
          label,
          type: 'sql',
          status: 'done',
          content: step?.sql || '',
          elapsedMs: typeof planMeta.elapsedMs === 'number' ? planMeta.elapsedMs : parsedFromNote.elapsedMs,
          performance: parsedFromNote.performance || null,
          optimization: parsedFromNote.optimization || null,
          note: fallbackNote,
        })
      })
    } else {
      // 支持旧的 sqls 数组（也兼容旧的 sql 字段）
      const sqls = m.sqls || (m.sql ? [m.sql] : [])
      sqls.forEach((sql, idx) => {
        const label = sqls.length > 1 ? `SQL 查询 ${idx + 1}` : 'SQL 查询'
        const planMeta = consumePlanMeta(sql) || {}
        const fallbackNote = planMeta.note || ''
        const parsedFromNote = parseSqlDisplayFromNote(fallbackNote)
        steps.push({
          key: `sql_${idx}`,
          label,
          type: 'sql',
          status: 'done',
          content: sql,
          elapsedMs: typeof planMeta.elapsedMs === 'number' ? planMeta.elapsedMs : parsedFromNote.elapsedMs,
          performance: parsedFromNote.performance,
          optimization: parsedFromNote.optimization,
          note: fallbackNote,
        })
      })
    }
    steps.push({ key: 'answer', label: '回答', type: 'answer', status: 'done', content: m.content })
    return { role: 'assistant', content: m.content, steps, streaming: false }
  })
  scrollToBottom()
}

async function submitQuery(text) {
  const query = (text ?? inputText.value).trim()
  if (!query || isStreaming.value) return

  // 如果当前没有活跃会话，先创建一个
  if (!props.threadId) {
    await ensureSession?.()
    // 等待 threadId prop 更新（下一个微任务）
    await new Promise(r => setTimeout(r, 0))
    if (!props.threadId) return  // 创建失败则放弃
  }

  inputText.value = ''
  isStreaming.value = true
  emit('query-start')
  messages.value.push({ role: 'user', content: query })

  // 添加 AI 占位消息
  messages.value.push({
    role: 'assistant',
    content: '',     // 历史加载 fallback 用
    steps: [],
    streaming: true,
  })
  const aiIdx = messages.value.length - 1

  // 更新会话名称（第一条消息作为标题）
  if (messages.value.filter(m => m.role === 'user').length === 1) {
    emit('session-named', { threadId: props.threadId, name: query.slice(0, 20) })
  }

  // 需要在时间线上显示的节点 → 普通步骤行
  const PLAIN_NODES = {
    intent_router: '分析意图',
    skill_router:  '技能路由',
  }
  // 技能执行节点 → 预插入 SQL 步骤（正在生成 SQL…）
  const SQL_SKILL_NODES = new Set(['simple_query', 'complex_query', 'data_analysis'])
  // 回答节点 → 预插入回答步骤（正在处理…）
  const ANSWER_NODES = new Set(['format_answer', 'general_chat'])

  let closeStream = null

  closeStream = streamQuery(query, props.threadId, props.sessionId, {
    onNodeStart(node) {
      const steps = messages.value[aiIdx].steps
      if (PLAIN_NODES[node]) {
        steps.push({ key: node, label: PLAIN_NODES[node], type: 'plain', status: 'running' })
      } else if (SQL_SKILL_NODES.has(node)) {
        steps.push({ key: 'sql_gen', label: '正在生成 SQL…', type: 'sql', status: 'pending', content: '' })
      } else if (ANSWER_NODES.has(node)) {
        steps.push({ key: 'answer', label: '正在处理…', type: 'answer', status: 'pending', content: '' })
      }
      scrollToBottom()
    },
    onNodeEnd(node) {
      const steps = messages.value[aiIdx].steps
      if (PLAIN_NODES[node]) {
        const idx = steps.findIndex(s => s.key === node)
        if (idx !== -1) steps.splice(idx, 1, { ...steps[idx], status: 'done' })
      }
    },
    async onSqlConfirm(sql, sessionId) {
      const steps = messages.value[aiIdx].steps
      // 找空槽（无内容的占位符），避免覆盖已确认的 SQL
      let sqlIdx = steps.findIndex(s => s.key === 'sql_gen' && !s.content)
      if (sqlIdx === -1) {
        // 没有空槽，追加新槽
        steps.push({ key: 'sql_gen', label: 'SQL 查询', type: 'sql', status: 'pending', content: '' })
        sqlIdx = steps.length - 1
      }
      steps.splice(sqlIdx, 1, {
        ...steps[sqlIdx],
        content: sql,
        label: 'SQL 查询',
        status: 'confirming',
      })
      scrollToBottom()

      const result = await openSqlConfirm?.(sql, sessionId)
      const isSkip = result?.action === 'skip'
      // 赋予唯一 key，防止后续调用再次找到并覆盖此槽
      steps.splice(sqlIdx, 1, {
        ...steps[sqlIdx],
        key: `sql_confirm_${Date.now()}_${sqlIdx}`,
        status: isSkip ? 'skipped' : 'done',
        label: isSkip ? 'SQL 查询（已跳过）' : 'SQL 查询',
      })
      return result
    },
    onSqlStep(stepId, label, sql, performance, optimization, elapsedMs) {
      // 每个 SQL 步骤执行后触发，为复杂查询/数据分析生成独立的 SQL 气泡
      const steps = messages.value[aiIdx].steps
      // 如果同一条 SQL 已在确认阶段展示过，则原位补齐执行结果与耗时。
      const existingIdx = steps.findIndex(s => s.type === 'sql' && s.content === sql)
      if (existingIdx !== -1) {
        steps.splice(existingIdx, 1, {
          ...steps[existingIdx],
          key: `sql_step_${stepId}`,
          label: label || steps[existingIdx].label || `SQL 查询 ${stepId}`,
          status: 'done',
          elapsedMs: typeof elapsedMs === 'number' ? elapsedMs : steps[existingIdx].elapsedMs ?? null,
          performance: performance || steps[existingIdx].performance || null,
          optimization: optimization || steps[existingIdx].optimization || null,
        })
        scrollToBottom()
        return
      }
      // 找第一个空的占位符（pending 且无内容）
      const emptyIdx = steps.findIndex(s => s.type === 'sql' && s.status === 'pending' && !s.content)
      const newStep = {
        key: `sql_step_${stepId}`,
        label: label || `SQL 查询 ${stepId}`,
        type: 'sql',
        status: 'done',
        content: sql,
        elapsedMs: typeof elapsedMs === 'number' ? elapsedMs : null,
        performance: performance || null,
        optimization: optimization || null,
      }
      if (emptyIdx !== -1) {
        steps.splice(emptyIdx, 1, newStep)
      } else {
        steps.push(newStep)
      }
      scrollToBottom()
    },
    onToken(content) {
      const steps = messages.value[aiIdx].steps
      const idx = steps.findIndex(s => s.key === 'answer')
      if (idx === -1) {
        steps.push({ key: 'answer', label: '回答', type: 'answer', status: 'streaming', content })
      } else {
        // splice 替换确保 Vue 3 响应式更新
        steps.splice(idx, 1, {
          ...steps[idx],
          label: '回答',
          status: 'streaming',
          content: (steps[idx].content || '') + content,
        })
      }
      scrollToBottom()
    },
    onFullResponse(content) {
      const steps = messages.value[aiIdx].steps
      const idx = steps.findIndex(s => s.key === 'answer')
      if (idx === -1) {
        steps.push({ key: 'answer', label: '回答', type: 'answer', status: 'streaming', content })
      } else if (!steps[idx].content) {
        steps.splice(idx, 1, {
          ...steps[idx],
          label: '回答',
          status: 'streaming',
          content,
        })
      }
      scrollToBottom()
    },
    onDone() {
      const steps = messages.value[aiIdx].steps
      const idx = steps.findIndex(s => s.key === 'answer')
      if (idx !== -1) {
        const answerContent = steps[idx].content || ''
        steps.splice(idx, 1, { ...steps[idx], status: 'done' })
        messages.value[aiIdx].content = answerContent
      }
      // 完成后只保留 sql 和 answer 步骤，隐藏路由等中间步骤
      const kept = messages.value[aiIdx].steps.filter(s => s.type === 'sql' || s.type === 'answer')
      messages.value[aiIdx].steps.splice(0, messages.value[aiIdx].steps.length, ...kept)
      messages.value[aiIdx].streaming = false
      isStreaming.value = false
      emit('query-done')
      scrollToBottom()
    },
    onError(message) {
      const steps = messages.value[aiIdx].steps
      steps.push({ key: 'error', label: '出错了', type: 'answer', status: 'done', content: `❌ ${message}` })
      messages.value[aiIdx].streaming = false
      isStreaming.value = false
      emit('query-done')
    },
  })
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

onMounted(() => {
  if (props.threadId) loadHistory(props.threadId)
})
</script>

<style scoped>
.chat-window {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f5f7fa;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px 16px 8px;
  scroll-behavior: smooth;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 16px;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  max-width: 480px;
}

.suggestion-tag {
  cursor: pointer;
  font-size: 13px;
  padding: 8px 14px;
  height: auto;
}

.input-area {
  padding: 12px 16px 16px;
  background: #fff;
  border-top: 1px solid #e8eaed;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.query-input {
  flex: 1;
}

.send-btn {
  flex-shrink: 0;
  height: 56px;
  width: 80px;
}
</style>
