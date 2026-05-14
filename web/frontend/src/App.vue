<template>
  <el-container class="app-container">
    <!-- 侧边栏：会话列表 -->
    <el-aside width="260px" class="sidebar">
      <div class="sidebar-header">
        <span class="sidebar-title">🤖 Text2SQL</span>
        <el-button
          type="primary"
          size="small"
          @click="newSession"
          :icon="Plus"
        >新对话</el-button>
      </div>

      <!-- 会话列表 -->
      <div class="session-list">
        <div
          v-for="session in sessions"
          :key="session.threadId"
          class="session-item"
          :class="{ active: session.threadId === currentThreadId }"
          @click="switchSession(session)"
        >
          <el-icon><ChatDotRound /></el-icon>
          <span class="session-name">{{ session.name }}</span>
          <el-icon class="session-delete" @click="deleteSessionItem(session, $event)" title="删除会话">
            <Delete />
          </el-icon>
        </div>
      </div>

      <div class="sidebar-footer">
        <el-tag type="info" size="small">FastAPI + LangGraph</el-tag>
      </div>
    </el-aside>

    <!-- 主区域：聊天窗口 -->
    <el-main class="main-area">
      <ChatWindow
        :thread-id="currentThreadId"
        :session-id="currentSessionId"
        @session-named="updateSessionName"
        @query-start="onQueryStart"
        @query-done="onQueryDone"
      />
    </el-main>
  </el-container>

  <!-- SQL 确认弹窗（全局）-->
  <SqlConfirmDialog ref="sqlConfirmDialogRef" />

  <!-- 任务链路 + 禁令抽屉 -->
  <PlanDrawer
    :thread-id="currentThreadId"
    :visible="drawerVisible"
    :polling="queryRunning"
    @toggle="drawerVisible = !drawerVisible"
  />
</template>

<script setup>
import { ref, onMounted, provide } from 'vue'
import { Plus, ChatDotRound, Delete } from '@element-plus/icons-vue'
import ChatWindow from './components/ChatWindow.vue'
import SqlConfirmDialog from './components/SqlConfirmDialog.vue'
import PlanDrawer from './components/PlanDrawer.vue'
import { createSession, listSessions, renameSession, deleteSession } from './api/chat.js'

const sessions = ref([])
const currentThreadId = ref('')
const currentSessionId = ref('')
const sqlConfirmDialogRef = ref(null)
const drawerVisible = ref(false)
const queryRunning = ref(false)

provide('openSqlConfirm', (sql, sessionId) => {
  return sqlConfirmDialogRef.value?.open(sql, sessionId)
})

function onQueryStart() { queryRunning.value = true }
function onQueryDone()  { queryRunning.value = false }

async function newSession() {
  const data = await createSession()
  const session = {
    threadId: data.thread_id,
    sessionId: generateSessionId(),
    name: data.name || '新对话',
  }
  sessions.value.unshift(session)
  currentThreadId.value = session.threadId
  currentSessionId.value = session.sessionId
}

function switchSession(session) {
  currentThreadId.value = session.threadId
  currentSessionId.value = session.sessionId
}

async function updateSessionName({ threadId, name }) {
  const s = sessions.value.find(s => s.threadId === threadId)
  if (s) s.name = name
  // 同步到后端 DB
  try { await renameSession(threadId, name) } catch { /* 非致命 */ }
}

function generateSessionId() {
  return Math.random().toString(36).slice(2)
}

async function deleteSessionItem(session, event) {
  event.stopPropagation()
  try {
    await deleteSession(session.threadId)
  } catch { /* 非致命 */ }
  const idx = sessions.value.findIndex(s => s.threadId === session.threadId)
  sessions.value.splice(idx, 1)
  if (session.threadId === currentThreadId.value) {
    if (sessions.value.length > 0) {
      const next = sessions.value[Math.min(idx, sessions.value.length - 1)]
      currentThreadId.value = next.threadId
      currentSessionId.value = next.sessionId
    } else {
      // 允许空状态，等用户提问时再创建
      currentThreadId.value = ''
      currentSessionId.value = ''
    }
  }
}

async function ensureSession() {
  if (currentThreadId.value) return
  const data = await createSession()
  const session = {
    threadId: data.thread_id,
    sessionId: generateSessionId(),
    name: data.name || '新对话',
  }
  sessions.value.unshift(session)
  currentThreadId.value = session.threadId
  currentSessionId.value = session.sessionId
}

provide('ensureSession', ensureSession)

onMounted(async () => {
  try {
    const { sessions: saved } = await listSessions()
    if (saved && saved.length > 0) {
      sessions.value = saved.map(s => ({
        threadId: s.thread_id,
        sessionId: generateSessionId(),
        name: s.name,
      }))
      // 激活最近会话
      const first = sessions.value[0]
      currentThreadId.value = first.threadId
      currentSessionId.value = first.sessionId
    }
    // 无历史会话时保持空状态，等用户提问再创建
  } catch { /* 服务器未准备好时保持空状态 */ }
})
</script>

<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
html, body, #app { height: 100%; }

.app-container {
  height: 100vh;
  background: #f5f7fa;
}

.sidebar {
  background: #1a1a2e;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-header {
  padding: 16px 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-title {
  color: #fff;
  font-size: 16px;
  font-weight: 600;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  color: rgba(255,255,255,0.7);
  transition: all 0.2s;
  margin-bottom: 2px;
}

.session-item:hover {
  background: rgba(255,255,255,0.1);
  color: #fff;
}

.session-item.active {
  background: rgba(64,158,255,0.3);
  color: #fff;
}

.session-name {
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.session-delete {
  opacity: 0;
  margin-left: auto;
  flex-shrink: 0;
  color: rgba(255,255,255,0.5);
  transition: opacity 0.2s, color 0.2s;
}

.session-item:hover .session-delete {
  opacity: 1;
}

.session-delete:hover {
  color: #f56c6c !important;
}

.sidebar-footer {
  padding: 12px;
  border-top: 1px solid rgba(255,255,255,0.1);
  display: flex;
  justify-content: center;
}

.main-area {
  padding: 0;
  overflow: hidden;
}
</style>
