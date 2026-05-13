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
import { streamQuery, getHistory } from '../api/chat.js'

const props = defineProps({
  threadId: { type: String, required: true },
  sessionId: { type: String, required: true },
})

const emit = defineEmits(['session-named', 'query-start', 'query-done'])

const openSqlConfirm = inject('openSqlConfirm')

const messages = ref([])
const inputText = ref('')
const isStreaming = ref(false)
const messagesContainer = ref(null)

const suggestions = [
  '数据库中有多少条记录？',
  '查看所有表名',
  '最近10条数据',
]

// 切换 session 时重新加载历史
watch(() => props.threadId, async (newId) => {
  if (newId) {
    messages.value = []
    await loadHistory(newId)
  }
})

async function loadHistory(threadId) {
  const { messages: hist } = await getHistory(threadId)
  messages.value = hist.map(m => {
    if (m.role !== 'assistant') return { role: m.role, content: m.content, steps: [], streaming: false }

    // 历史 AI 消息重建为 steps 格式，与流式完成后保持一致
    const steps = []
    if (m.sql) {
      steps.push({ key: 'sql_gen', label: 'SQL 查询', type: 'sql', status: 'done', content: m.sql })
    }
    steps.push({ key: 'answer', label: '回答', type: 'answer', status: 'done', content: m.content })
    return { role: 'assistant', content: m.content, steps, streaming: false }
  })
  scrollToBottom()
}

async function submitQuery(text) {
  const query = (text ?? inputText.value).trim()
  if (!query || isStreaming.value) return

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
      let sqlIdx = steps.findIndex(s => s.key === 'sql_gen')
      if (sqlIdx === -1) {
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
      steps.splice(sqlIdx, 1, {
        ...steps[sqlIdx],
        status: isSkip ? 'skipped' : 'done',
        label: isSkip ? 'SQL 查询（已跳过）' : 'SQL 查询',
      })
      return result
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
