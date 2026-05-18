<template>
  <div class="placeholder-view">
    <div class="placeholder-content">
      <div class="placeholder-icon">🚧</div>
      <h2 class="placeholder-title">{{ title }}</h2>
      <p class="placeholder-desc">功能正在开发中，敬请期待</p>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: '功能开发中',
  },
})
</script>

<style scoped>
.placeholder-view {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  background: #f5f7fa;
}

.placeholder-content {
  text-align: center;
  color: #909399;
}

.placeholder-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.placeholder-title {
  font-size: 22px;
  font-weight: 600;
  margin: 0 0 8px;
  color: #606266;
}

.placeholder-desc {
  font-size: 14px;
  margin: 0;
}
</style>
