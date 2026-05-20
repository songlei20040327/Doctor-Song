<template>
    <div class="settings-panel">
        <div class="setting-group">
            <label class="setting-label">Temperature</label>
            <div class="setting-row">
                <el-slider
                    v-model="localSettings.temperature"
                    :min="0"
                    :max="2"
                    :step="0.1"
                    :marks="{ 0: '0', 0.5: '0.5', 1: '1', 1.5: '1.5', 2: '2' }"
                    show-input
                />
            </div>
            <p class="setting-hint">越高越有创意，越低越保守精确</p>
        </div>

        <div class="setting-group">
            <label class="setting-label">最大输出长度</label>
            <div class="setting-row">
                <el-slider
                    v-model="localSettings.maxLength"
                    :min="256"
                    :max="4096"
                    :step="128"
                    show-input
                />
            </div>
            <p class="setting-hint">控制单次回复的最大 token 数</p>
        </div>

        <div class="setting-group">
            <label class="setting-label">System Prompt</label>
            <el-input
                v-model="localSettings.systemPrompt"
                type="textarea"
                :rows="4"
                placeholder="设定AI助手的角色与行为..."
            />
            <p class="setting-hint">自定义系统提示词，引导AI的回复风格</p>
        </div>

        <div class="setting-actions">
            <el-button type="primary" @click="saveSettings" round>保存设置</el-button>
            <el-button @click="resetDefaults" round>恢复默认</el-button>
        </div>
    </div>
</template>

<script setup>
import { reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'

const emit = defineEmits(['close'])

const DEFAULT_SETTINGS = {
    temperature: 0.7,
    maxLength: 2048,
    systemPrompt: '你是 Doctor.Song，一位专业的医学AI助手。请给出准确、安全、循证的医学建议。',
}

const STORAGE_KEY = 'doctor_song_settings'

const localSettings = reactive({ ...DEFAULT_SETTINGS })

const loadSettings = () => {
    try {
        const saved = localStorage.getItem(STORAGE_KEY)
        if (saved) {
            Object.assign(localSettings, JSON.parse(saved))
        }
    } catch (e) { /* ignore */ }
}

const saveSettings = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...localSettings }))
    ElMessage.success('设置已保存')
    emit('close')
}

const resetDefaults = () => {
    Object.assign(localSettings, DEFAULT_SETTINGS)
    localStorage.removeItem(STORAGE_KEY)
    ElMessage.success('已恢复默认设置')
}

onMounted(loadSettings)
</script>

<style scoped>
.settings-panel {
    padding: 4px 0;
}

.setting-group {
    margin-bottom: 28px;
}

.setting-label {
    display: block;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
}

.setting-row {
    padding: 0 4px;
}

.setting-hint {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 6px;
    line-height: 1.5;
}

.setting-actions {
    display: flex;
    gap: 12px;
    margin-top: 32px;
}
</style>
