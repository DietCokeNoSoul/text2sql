<template>
  <div class="constraints-panel">
    <div class="cp-header">
      <span class="cp-title">⛔ 禁令管理</span>
      <el-tooltip content="禁令会在每次 LLM 调用时注入系统提示词，始终生效，不受记忆压缩影响" placement="bottom">
        <el-icon class="cp-help"><QuestionFilled /></el-icon>
      </el-tooltip>
    </div>

    <!-- 禁令列表 -->
    <div class="cp-list" v-if="constraints.length > 0">
      <div
        v-for="c in constraints"
        :key="c.id"
        class="cp-item"
        :class="{ disabled: !c.enabled }"
      >
        <el-switch
          :model-value="!!c.enabled"
          size="small"
          @change="(val) => toggle(c, val)"
          class="cp-switch"
        />
        <span class="cp-content">{{ c.content }}</span>
        <el-button
          type="danger"
          text
          size="small"
          @click="remove(c.id)"
          class="cp-del"
        >
          <el-icon><Delete /></el-icon>
        </el-button>
      </div>
    </div>
    <div v-else class="cp-empty">暂无禁令<br /><small>添加后每轮对话均会强制执行</small></div>

    <!-- 输入区 -->
    <div class="cp-input-row">
      <el-input
        v-model="newContent"
        placeholder="输入禁令，如：不得查询 salary 字段"
        size="small"
        :disabled="adding"
        @keydown.enter.exact.prevent="add"
        clearable
        class="cp-input"
      />
      <el-button
        type="primary"
        size="small"
        :loading="adding"
        :disabled="!newContent.trim()"
        @click="add"
      >添加</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { Delete, QuestionFilled } from '@element-plus/icons-vue'
import { listConstraints, addConstraint, toggleConstraint, deleteConstraint } from '../api/chat.js'
import { ElMessage } from 'element-plus'

const props = defineProps({
  threadId: { type: String, required: true },
})

const emit = defineEmits(['count-change'])

const constraints = ref([])
const newContent = ref('')
const adding = ref(false)

function emitCount() {
  emit('count-change', constraints.value.filter(c => c.enabled).length)
}

async function load() {
  if (!props.threadId) return
  try {
    const { constraints: list } = await listConstraints(props.threadId)
    constraints.value = list
    emitCount()
  } catch {
    constraints.value = []
  }
}

async function add() {
  const content = newContent.value.trim()
  if (!content) return
  adding.value = true
  try {
    const item = await addConstraint(props.threadId, content)
    constraints.value.push(item)
    newContent.value = ''
    emitCount()
  } catch {
    ElMessage.error('添加失败')
  } finally {
    adding.value = false
  }
}

async function toggle(c, val) {
  try {
    await toggleConstraint(props.threadId, c.id, val)
    c.enabled = val ? 1 : 0
    emitCount()
  } catch {
    ElMessage.error('切换失败')
  }
}

async function remove(id) {
  try {
    await deleteConstraint(props.threadId, id)
    constraints.value = constraints.value.filter(c => c.id !== id)
    emitCount()
  } catch {
    ElMessage.error('删除失败')
  }
}

watch(() => props.threadId, load)
onMounted(load)
</script>

<style scoped>
.constraints-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 8px 12px 12px;
}

.cp-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 0 10px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  margin-bottom: 8px;
}

.cp-title {
  color: var(--el-text-color-primary);
  font-size: 13px;
  font-weight: 600;
}

.cp-help {
  color: var(--el-text-color-placeholder);
  cursor: help;
  font-size: 14px;
}

.cp-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 10px;
}

.cp-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 10px;
  background: var(--el-fill-color-lighter);
  border-radius: 8px;
  border: 1px solid var(--el-color-danger-light-7);
  transition: opacity 0.2s;
}

.cp-item.disabled {
  opacity: 0.45;
}

.cp-switch {
  flex-shrink: 0;
  margin-top: 2px;
}

.cp-content {
  flex: 1;
  font-size: 12px;
  color: var(--el-text-color-primary);
  line-height: 1.5;
  word-break: break-all;
}

.cp-del {
  flex-shrink: 0;
  padding: 0 2px !important;
}

.cp-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--el-text-color-placeholder);
  font-size: 13px;
  line-height: 1.8;
  margin-bottom: 10px;
}

.cp-input-row {
  display: flex;
  gap: 6px;
  align-items: center;
}

.cp-input {
  flex: 1;
}
</style>
