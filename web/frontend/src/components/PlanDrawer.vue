<template>
  <!-- Toggle button pinned to right edge -->
  <div class="plan-toggle-btn" :class="{ open: visible }" @click="$emit('toggle')">
    <el-icon><ArrowLeft v-if="visible" /><ArrowRight v-else /></el-icon>
    <span v-if="!visible && constraintCount > 0" class="toggle-badge">{{ constraintCount }}</span>
  </div>

  <!-- Sliding drawer -->
  <transition name="drawer-slide">
    <div v-if="visible" class="plan-drawer">
      <!-- Drawer header with tab switcher -->
      <div class="drawer-header">
        <div class="drawer-tabs">
          <button
            class="drawer-tab"
            :class="{ active: drawerTab === 'plans' }"
            @click="drawerTab = 'plans'"
          >📋 任务链路</button>
          <button
            class="drawer-tab"
            :class="{ active: drawerTab === 'constraints' }"
            @click="drawerTab = 'constraints'"
          >
            ⛔ 禁令
            <span v-if="constraintCount > 0" class="tab-badge">{{ constraintCount }}</span>
          </button>
        </div>
        <el-button
          v-if="drawerTab === 'plans'"
          text size="small"
          @click="refresh"
          :loading="loading"
        >
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>

      <!-- 任务链路 tab -->
      <template v-if="drawerTab === 'plans'">
        <div class="drawer-body" v-if="plans.length > 0">
          <div v-for="plan in plans" :key="plan.task_id" class="plan-card">
            <div class="plan-card-header">
              <span class="plan-skill-tag" :class="plan.skill">{{ skillLabel(plan.skill) }}</span>
              <span class="plan-status" :class="plan.status">{{ statusLabel(plan.status) }}</span>
            </div>
            <div class="plan-title" :title="plan.title">{{ plan.title }}</div>
            <div class="plan-time">{{ plan.created_at }}</div>

            <div class="plan-steps">
              <div
                v-for="step in plan.steps"
                :key="step.step_id"
                class="plan-step"
                :class="step.status"
              >
                <span class="step-icon">{{ stepIcon(step.status) }}</span>
                <div class="step-content">
                  <span class="step-desc">{{ step.description }}</span>
                  <div v-if="step.notes" class="step-note">{{ step.notes }}</div>
                </div>
              </div>
            </div>

            <div v-if="lastSummary(plan)" class="plan-result">
              <span class="result-label">结果摘要：</span>{{ lastSummary(plan) }}
            </div>
          </div>
        </div>
        <div v-else-if="loading" class="drawer-empty">加载中…</div>
        <div v-else class="drawer-empty">
          此会话暂无任务链路<br />
          <small>每次查询时自动生成</small>
        </div>
      </template>

      <!-- 禁令 tab -->
      <ConstraintsPanel
        v-else
        :thread-id="threadId"
        @count-change="onConstraintCountChange"
        class="drawer-constraints"
      />
    </div>
  </transition>
</template>

<script setup>
import { ref, watch, onUnmounted } from 'vue'
import { ArrowLeft, ArrowRight, Refresh } from '@element-plus/icons-vue'
import { getPlans } from '../api/chat.js'
import ConstraintsPanel from './ConstraintsPanel.vue'

const props = defineProps({
  threadId: { type: String, default: '' },
  visible:  { type: Boolean, default: false },
  polling:  { type: Boolean, default: false },
})
defineEmits(['toggle'])

const drawerTab = ref('plans')
const constraintCount = ref(0)

function onConstraintCountChange(count) {
  constraintCount.value = count
}

// Reset tab to plans when thread changes
watch(() => props.threadId, () => {
  constraintCount.value = 0
})

const plans   = ref([])
const loading = ref(false)
let   _timer  = null

async function refresh() {
  if (!props.threadId) return
  loading.value = true
  try {
    const data = await getPlans(props.threadId)
    plans.value = data.plans || []
  } finally {
    loading.value = false
  }
}

function startPolling() {
  stopPolling()
  _timer = setInterval(refresh, 2000)
}
function stopPolling() {
  if (_timer) { clearInterval(_timer); _timer = null }
}

// Reload whenever drawer opens or thread changes
watch(() => [props.visible, props.threadId], ([vis]) => {
  if (vis && props.threadId) refresh()
}, { immediate: true })

// Poll while a query is running
watch(() => props.polling, (v) => {
  if (v) startPolling()
  else   { stopPolling(); if (props.visible) refresh() }
})

onUnmounted(stopPolling)

// ── helpers ──────────────────────────────────────────────────────────────────

function skillLabel(skill) {
  return { simple_query: '简单查询', complex_query: '复杂查询', data_analysis: '数据分析' }[skill] ?? skill
}
function statusLabel(status) {
  return { in_progress: '进行中', done: '已完成', failed: '失败', pending: '待执行' }[status] ?? status
}
function stepIcon(status) {
  return { done: '✓', in_progress: '⟳', failed: '✗', pending: '·', skipped: '—' }[status] ?? '·'
}
function lastSummary(plan) {
  const done = (plan.steps || []).filter(s => s.status === 'done' && s.result_summary)
  return done.length ? done[done.length - 1].result_summary : ''
}
</script>

<style scoped>
/* ── Toggle button ───────────────────────────────────────────────────────── */
.plan-toggle-btn {
  position: fixed;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  z-index: 1001;
  width: 24px;
  height: 56px;
  background: var(--el-color-primary);
  color: #fff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: 6px 0 0 6px;
  cursor: pointer;
  box-shadow: -2px 0 8px rgba(0,0,0,.15);
  transition: background .2s;
  font-size: 14px;
}
.plan-toggle-btn:hover { background: var(--el-color-primary-dark-2); }
.plan-toggle-btn.open  { right: 320px; }

.toggle-badge {
  background: #ff4d4f;
  color: #fff;
  border-radius: 8px;
  font-size: 9px;
  padding: 0 3px;
  min-width: 14px;
  text-align: center;
  line-height: 14px;
  margin-top: 2px;
}

/* ── Drawer panel ────────────────────────────────────────────────────────── */
.plan-drawer {
  position: fixed;
  right: 0;
  top: 0;
  bottom: 0;
  width: 320px;
  background: var(--el-bg-color);
  border-left: 1px solid var(--el-border-color-light);
  box-shadow: -4px 0 16px rgba(0,0,0,.08);
  z-index: 1000;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Slide animation ─────────────────────────────────────────────────────── */
.drawer-slide-enter-active,
.drawer-slide-leave-active { transition: transform .25s ease; }
.drawer-slide-enter-from,
.drawer-slide-leave-to     { transform: translateX(100%); }

/* ── Header ──────────────────────────────────────────────────────────────── */
.drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);
  flex-shrink: 0;
  gap: 8px;
}
.drawer-title { font-weight: 600; font-size: 14px; }

/* ── Drawer tabs ─────────────────────────────────────────────────────────── */
.drawer-tabs {
  display: flex;
  gap: 4px;
  flex: 1;
}

.drawer-tab {
  flex: 1;
  padding: 6px 8px;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all .2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  white-space: nowrap;
}
.drawer-tab:hover { color: var(--el-color-primary); }
.drawer-tab.active {
  color: var(--el-color-primary);
  border-bottom-color: var(--el-color-primary);
  font-weight: 600;
}

.tab-badge {
  background: #ff4d4f;
  color: #fff;
  border-radius: 10px;
  font-size: 10px;
  padding: 0 5px;
  min-width: 16px;
  text-align: center;
  line-height: 16px;
}

/* ── Constraints panel inside drawer ─────────────────────────────────────── */
.drawer-constraints {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── Body ────────────────────────────────────────────────────────────────── */
.drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ── Plan card ───────────────────────────────────────────────────────────── */
.plan-card {
  background: var(--el-fill-color-lighter);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 12px;
}
.plan-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}
.plan-skill-tag {
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  background: var(--el-color-primary-light-8);
  color: var(--el-color-primary);
}
.plan-skill-tag.data_analysis {
  background: var(--el-color-success-light-8);
  color: var(--el-color-success);
}
.plan-status {
  font-size: 11px;
  margin-left: auto;
}
.plan-status.done        { color: var(--el-color-success); }
.plan-status.in_progress { color: var(--el-color-warning); }
.plan-status.failed      { color: var(--el-color-danger); }

.plan-title {
  font-weight: 500;
  margin-bottom: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.plan-time { color: var(--el-text-color-secondary); font-size: 11px; margin-bottom: 8px; }

/* ── Steps ───────────────────────────────────────────────────────────────── */
.plan-steps { display: flex; flex-direction: column; gap: 3px; }
.plan-step {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  color: var(--el-text-color-secondary);
}
.plan-step.done        { color: var(--el-color-success); }
.plan-step.in_progress { color: var(--el-color-warning); }
.plan-step.failed      { color: var(--el-color-danger); }

.step-icon  { width: 14px; text-align: center; flex-shrink: 0; }
.step-content { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.step-desc  { font-size: 11px; line-height: 1.4; }
.step-note {
  font-size: 10px;
  line-height: 1.35;
  color: var(--el-text-color-secondary);
  opacity: .92;
  word-break: break-word;
}

/* ── Result summary ──────────────────────────────────────────────────────── */
.plan-result {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid var(--el-border-color-lighter);
  font-size: 11px;
  color: var(--el-text-color-secondary);
  word-break: break-all;
}
.result-label { font-weight: 500; color: var(--el-text-color-primary); }

/* ── Empty state ─────────────────────────────────────────────────────────── */
.drawer-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--el-text-color-secondary);
  font-size: 13px;
  text-align: center;
  line-height: 1.8;
  padding: 24px;
}
</style>
