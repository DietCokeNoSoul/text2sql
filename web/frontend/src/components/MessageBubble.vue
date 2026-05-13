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
        <div
          v-for="(step, i) in message.steps"
          :key="i"
          class="step-row"
          :class="'type-' + step.type"
        >
          <!-- 普通路由步骤（流式中才显示，完成后已被过滤） -->
          <template v-if="step.type === 'plain'">
            <span class="step-label-plain">{{ step.label }}</span>
            <span v-if="step.status === 'done'" class="step-check-inline">✓</span>
            <span v-else class="step-dots-ani"><span/><span/><span/></span>
          </template>

          <!-- SQL 步骤 -->
          <template v-else-if="step.type === 'sql'">
            <div class="step-header-row">
              <span class="step-section-label">{{ step.label }}</span>
              <span v-if="step.status === 'done'" class="step-check">✓</span>
              <span v-else-if="step.status === 'skipped'" class="step-skip">跳过</span>
              <span v-else class="step-dots-ani"><span/><span/><span/></span>
            </div>
            <div v-if="step.content" class="sql-block">
              <pre><code>{{ step.content }}</code></pre>
            </div>
          </template>

          <!-- 回答步骤 -->
          <template v-else-if="step.type === 'answer'">
            <div class="step-header-row">
              <span class="step-section-label">{{ step.content ? '回答' : '正在处理…' }}</span>
              <span v-if="step.status === 'streaming'" class="cursor">▍</span>
            </div>
            <div v-if="step.content" class="answer-content markdown-body">
              <span v-html="renderMarkdown(step.content)" />
            </div>
            <div v-else-if="step.status !== 'done'" class="inline-wait">
              <span class="step-dots-ani"><span/><span/><span/></span>
            </div>
          </template>
        </div>
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
import { marked } from 'marked'

defineProps({
  message: { type: Object, required: true },
})

function renderMarkdown(text) {
  return marked.parse(text ?? '')
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
