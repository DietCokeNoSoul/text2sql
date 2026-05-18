<template>
  <div class="app-root">
    <!-- 顶部导航栏 -->
    <div class="top-nav">
      <div class="nav-brand">🤖 Text2SQL</div>
      <div class="nav-tabs">
        <div
          v-for="tab in tabs"
          :key="tab.key"
          class="nav-tab"
          :class="{ active: currentTab === tab.key }"
          @click="currentTab = tab.key"
        >
          <span class="tab-icon">{{ tab.icon }}</span>
          <span class="tab-label">{{ tab.label }}</span>
        </div>
      </div>
    </div>

    <!-- 内容区 -->
    <div class="content-area">
      <!-- 对话 -->
      <template v-if="currentTab === 'chat'">
        <el-aside width="260px" class="sidebar">
          <div class="sidebar-header">
            <el-button
              type="primary"
              size="small"
              @click="newSession"
              :icon="Plus"
            >新对话</el-button>
          </div>
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
        <div class="chat-main">
          <ChatWindow
            :thread-id="currentThreadId"
            :session-id="currentSessionId"
            @session-named="updateSessionName"
            @query-start="onQueryStart"
            @query-done="onQueryDone"
          />
        </div>
      </template>

      <!-- 统计 -->
      <StatsView v-else-if="currentTab === 'stats'" />

      <!-- 其他（占位） -->
      <PlaceholderView
        v-else
        :title="tabs.find(t => t.key === currentTab)?.label"
      />
    </div>
  </div>

  <!-- SQL 确认弹窗（全局，仅对话页使用）-->
  <SqlConfirmDialog ref="sqlConfirmDialogRef" />

  <!-- 任务链路 + 禁令抽屉 -->
  <PlanDrawer
    v-if="currentTab === 'chat'"
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
import StatsView from './views/StatsView.vue'
import PlaceholderView from './views/PlaceholderView.vue'
import { createSession, listSessions, renameSession, deleteSession } from './api/chat.js'

// ── 导航 ─────────────────────────────────────────────────────────────────────
const currentTab = ref('chat')
const tabs = [
  { key: 'chat',    icon: '💬', label: '对话' },
  { key: 'query',   icon: '🔍', label: '查询' },
  { key: 'schema',  icon: '📋', label: '表结构变更' },
  { key: 'monitor', icon: '📊', label: '监控告警' },
  { key: 'slow',    icon: '🐢', label: '慢日志' },
  { key: 'stats',   icon: '📈', label: '统计' },
]

// ── 会话管理 ──────────────────────────────────────────────────────────────────
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
  try { await renameSession(threadId, name) } catch { /* 非致命 */ }
}

function generateSessionId() {
  return Math.random().toString(36).slice(2)
}

async function deleteSessionItem(session, event) {
  event.stopPropagation()
  try { await deleteSession(session.threadId) } catch { /* 非致命 */ }
  const idx = sessions.value.findIndex(s => s.threadId === session.threadId)
  sessions.value.splice(idx, 1)
  if (session.threadId === currentThreadId.value) {
    if (sessions.value.length > 0) {
      const next = sessions.value[Math.min(idx, sessions.value.length - 1)]
      currentThreadId.value = next.threadId
      currentSessionId.value = next.sessionId
    } else {
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
      const first = sessions.value[0]
      currentThreadId.value = first.threadId
      currentSessionId.value = first.sessionId
    }
  } catch { /* 服务器未准备好时保持空状态 */ }
})
</script>

<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
html, body, #app { height: 100%; }

/* ── 顶层布局 ── */
.app-root {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── 顶部导航栏 ── */
.top-nav {
  height: 52px;
  flex-shrink: 0;
  background: #1a1a2e;
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 24px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  z-index: 100;
}

.nav-brand {
  color: #fff;
  font-size: 16px;
  font-weight: 700;
  white-space: nowrap;
  flex-shrink: 0;
}

.nav-tabs {
  display: flex;
  gap: 4px;
  flex: 1;
}

.nav-tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 16px;
  border-radius: 6px;
  cursor: pointer;
  color: rgba(255,255,255,0.6);
  font-size: 14px;
  transition: all 0.2s;
  user-select: none;
  white-space: nowrap;
}

.nav-tab:hover {
  background: rgba(255,255,255,0.1);
  color: rgba(255,255,255,0.9);
}

.nav-tab.active {
  background: rgba(64,158,255,0.3);
  color: #fff;
  font-weight: 600;
}

.tab-icon { font-size: 15px; }

/* ── 内容区 ── */
.content-area {
  flex: 1;
  overflow: hidden;
  display: flex;
}

/* ── 对话侧边栏 ── */
.sidebar {
  background: #1a1a2e;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex-shrink: 0;
}

.sidebar-header {
  padding: 12px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  border-bottom: 1px solid rgba(255,255,255,0.1);
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

.session-item:hover .session-delete { opacity: 1; }
.session-delete:hover { color: #f56c6c !important; }

.sidebar-footer {
  padding: 12px;
  border-top: 1px solid rgba(255,255,255,0.1);
  display: flex;
  justify-content: center;
}

/* ── 聊天主区 ── */
.chat-main {
  flex: 1;
  overflow: hidden;
}
</style>

