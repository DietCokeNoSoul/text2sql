<template>
  <el-dialog
    v-model="visible"
    title="⚠️ SQL 执行确认"
    width="580px"
    :close-on-click-modal="false"
    :close-on-press-escape="false"
    :show-close="false"
    align-center
    class="sql-confirm-dialog"
  >
    <div class="sql-block">
      <div class="sql-label">待执行 SQL：</div>
      <pre class="sql-code">{{ currentSql }}</pre>
    </div>

    <div v-if="showReasonInput" class="reason-input">
      <el-input
        v-model="skipReason"
        placeholder="跳过原因（可选）"
        clearable
      />
    </div>

    <template #footer>
      <div class="dialog-footer">
        <el-button
          type="danger"
          plain
          @click="handleSkip"
          :disabled="resolving"
        >
          ✕ 跳过此 SQL
        </el-button>
        <el-button
          type="primary"
          @click="handleExecute"
          :loading="resolving"
        >
          ▶ 执行
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref } from 'vue'

const visible = ref(false)
const currentSql = ref('')
const currentSessionId = ref('')
const showReasonInput = ref(false)
const skipReason = ref('')
const resolving = ref(false)

let _resolve = null

/**
 * Open the dialog and return a Promise that resolves with {action, reason}.
 * Called by the parent (App.vue) via template ref.
 */
function open(sql, sessionId) {
  currentSql.value = sql
  currentSessionId.value = sessionId
  skipReason.value = ''
  showReasonInput.value = false
  resolving.value = false
  visible.value = true

  return new Promise((resolve) => {
    _resolve = resolve
  })
}

function handleExecute() {
  resolving.value = true
  visible.value = false
  _resolve?.({ action: 'execute', reason: '' })
  _resolve = null
}

function handleSkip() {
  if (!showReasonInput.value) {
    // 第一次点跳过：显示原因输入框
    showReasonInput.value = true
    return
  }
  visible.value = false
  _resolve?.({ action: 'skip', reason: skipReason.value })
  _resolve = null
}

defineExpose({ open })
</script>

<style scoped>
.sql-block {
  background: #1e1e2e;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}

.sql-label {
  color: #a6adc8;
  font-size: 12px;
  margin-bottom: 8px;
}

.sql-code {
  margin: 0;
  color: #cdd6f4;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-all;
}

.reason-input {
  margin-top: 8px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}
</style>
