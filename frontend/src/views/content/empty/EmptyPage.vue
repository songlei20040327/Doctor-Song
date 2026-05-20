<template>
    <div class="empty-page">
        <el-icon :size="48" color="var(--text-muted)"><Clock /></el-icon>
        <p class="empty-text">功能开发中，敬请期待...</p>
    </div>
</template>

<script setup>
import { Clock } from '@element-plus/icons-vue'
</script>

<style scoped>
.empty-page {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
    background: var(--bg-body);
    transition: background-color 0.3s ease;
}

.empty-text {
    font-size: 16px;
    color: var(--text-muted);
}
</style>
