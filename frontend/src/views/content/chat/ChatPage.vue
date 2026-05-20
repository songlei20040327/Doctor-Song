<template>
    <div class="chat-container">
        <!-- left history panel -->
        <div class="history-panel">
            <button class="new-chat-btn" @click="newConversation">
                <el-icon :size="16"><Plus/></el-icon>
                <span>新建对话</span>
            </button>

            <div class="history-list" v-if="historyList.length > 0">
                <div
                    v-for="(item, index) in historyList"
                    :key="index"
                    class="history-item"
                    :class="{ active: currentConversationIndex === index }"
                    @click="selectConversation(index)"
                >
                    <div class="history-item-content">
                        <div class="history-title" v-if="editingIndex !== index">
                            {{ item.title }}
                        </div>
                        <input
                            v-else
                            class="history-title-input"
                            v-model="editTitle"
                            @blur="finishEdit(index)"
                            @keyup.enter="finishEdit(index)"
                            @click.stop
                            ref="titleInputRef"
                        />
                        <div class="history-preview">{{ getPreview(item) }}</div>
                        <div class="history-time">{{ formatTime(item.updatedAt) }}</div>
                    </div>
                    <div class="history-actions">
                        <el-button
                            class="action-btn"
                            circle
                            size="small"
                            @click.stop="startEdit(index, item.title)"
                            title="重命名"
                        >
                            <el-icon :size="14"><EditPen /></el-icon>
                        </el-button>
                        <el-popconfirm
                            title="确定删除此对话？"
                            @confirm="deleteConversation(index)"
                            confirm-button-text="删除"
                            cancel-button-text="取消"
                        >
                            <template #reference>
                                <el-button
                                    class="action-btn"
                                    circle
                                    size="small"
                                    @click.stop
                                    title="删除"
                                >
                                    <el-icon :size="14"><Delete /></el-icon>
                                </el-button>
                            </template>
                        </el-popconfirm>
                    </div>
                </div>
            </div>

            <div class="history-empty" v-else>
                <p>暂无对话记录</p>
            </div>
        </div>

        <!-- right chat panel -->
        <div class="chat-wrapper">
            <div class="chat-panel">
                <div class="chat-messages" ref="chatMessagesRef">
                    <div v-for="(message, index) in currentConversation.messages"
                         :key="index" :class="['message', message.role]">
                        <div class="avatar">
                            <template v-if="message.role === 'assistant'">
                                <img src="@/assets/sunli.png" alt="Doctor.Song">
                            </template>
                            <template v-else>
                                <img src="@/assets/dengchao.png" alt="Me">
                            </template>
                        </div>
                        <div class="bubble">
                            <!-- thinking block -->
                            <div v-if="message.thinkText" class="think-block">
                                <div class="think-header" @click="toggleThink(index)">
                                    <el-icon class="think-icon">
                                        <component :is="message.thinkOpen ? 'ArrowDown' : 'ArrowRight'" />
                                    </el-icon>
                                    <span>{{ message.thinkDone ? '思考过程' : '正在思考...' }}</span>
                                    <span v-if="message.thinkTime !== undefined && message.thinkTime !== null" class="think-time">{{ message.thinkTime }}s</span>
                                    <span v-if="!message.thinkDone" class="think-spinner"></span>
                                </div>
                                <div v-show="message.thinkOpen" class="think-content">
                                    {{ message.thinkText }}
                                </div>
                            </div>
                            <!-- answer text with markdown -->
                            <div v-if="message.content" class="answer-text" v-html="renderMarkdown(message.content)"></div>
                            <!-- loading dots -->
                            <div v-if="message.loading && !message.content && !message.thinkText" class="loading-dots">
                                <span></span><span></span><span></span>
                            </div>
                            <!-- streaming cursor -->
                            <span v-if="message.streaming" class="stream-cursor">|</span>
                        </div>
                    </div>
                    <div v-if="currentConversation.messages.length === 0" class="welcome-hint">
                        <img src="@/assets/sunli.png" alt="Doctor.Song" class="welcome-logo" />
                        <h2>你好，我是 Doctor.Song</h2>
                        <p>专业的医学AI助手，基于循证医学知识为您提供参考建议</p>
                        <p class="welcome-disclaimer">AI 生成的医学建议仅供参考，不能替代专业医生诊断</p>
                    </div>
                </div>

                <!-- input area -->
                <div class="input-area">
                    <div class="input-wrapper">
                        <input
                            v-model="userInput"
                            @keyup.enter="sendMessage"
                            placeholder="输入您的问题，按回车发送..."
                            type="text"
                            :disabled="isLoading"
                        >
                        <div class="button-group">
                            <button
                                v-if="isLoading"
                                class="circle-btn stop-btn"
                                @click="stopStreaming"
                                title="停止生成"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
                            </button>
                            <button
                                v-else
                                class="circle-btn send-btn"
                                @click="sendMessage"
                                :disabled="!userInput.trim()"
                                title="发送"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, computed, nextTick, watch, onMounted } from 'vue'
import { Plus, ArrowDown, ArrowRight, EditPen, Delete } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'
import { API, BASE_URL } from '@/api/config'

// --- Markdown renderer ---
const md = new MarkdownIt({
    html: false,
    breaks: true,
    linkify: true,
    typographer: true,
    highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return '<pre class="hljs"><code>' +
                    hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
                    '</code></pre>'
            } catch (e) { /* ignore */ }
        }
        return '<pre class="hljs"><code>' + md.utils.escapeHtml(str) + '</code></pre>'
    }
})

const renderMarkdown = (text) => {
    if (!text) return ''
    return md.render(text)
}

// --- Settings ---
const STORAGE_KEY = 'doctor_song_chat_history'
const SETTINGS_KEY = 'doctor_song_settings'

const getSettings = () => {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY)
        if (saved) return JSON.parse(saved)
    } catch (e) { /* ignore */ }
    return {
        temperature: 0.7,
        maxLength: 2048,
        systemPrompt: '你是 Doctor.Song，一位专业的医学AI助手。请给出准确、安全、循证的医学建议。',
    }
}

const loadHistory = () => {
    try {
        const saved = localStorage.getItem(STORAGE_KEY)
        if (saved) {
            const parsed = JSON.parse(saved)
            if (Array.isArray(parsed) && parsed.length > 0) {
                for (const conv of parsed) {
                    for (const msg of conv.messages) {
                        if (msg.thinkText === undefined) msg.thinkText = ''
                        if (msg.thinkOpen === undefined) msg.thinkOpen = true
                        if (msg.thinkDone === undefined) msg.thinkDone = true
                        if (msg.streaming === undefined) msg.streaming = false
                        if (msg.thinkTime === undefined) msg.thinkTime = null
                    }
                    if (!conv.updatedAt) conv.updatedAt = Date.now()
                }
                return parsed
            }
        }
    } catch (e) { /* ignore */ }
    return [{
        title: '新对话',
        updatedAt: Date.now(),
        messages: [{
            role: 'assistant',
            content: '你好！我是 Doctor.Song，你的专属医学AI助手。我学习了大量医学知识，可以回答你的健康问题。有什么我可以帮助你的吗？',
            thinkText: '',
            thinkOpen: true,
            thinkDone: true,
            streaming: false,
            thinkTime: null,
        }]
    }]
}

// --- State ---
let activeAbortCtrl = null
const historyList = ref(loadHistory())
const currentConversationIndex = ref(0)
const userInput = ref('')
const isLoading = ref(false)
const chatMessagesRef = ref(null)
const editingIndex = ref(-1)
const editTitle = ref('')
const titleInputRef = ref(null)

const currentConversation = computed(() => historyList.value[currentConversationIndex.value])

// persist
watch(historyList, (newVal) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newVal))
}, { deep: true })

const scrollToBottom = () => {
    nextTick(() => {
        const el = chatMessagesRef.value
        if (el) el.scrollTop = el.scrollHeight
    })
}

const selectConversation = (index) => {
    currentConversationIndex.value = index
    editingIndex.value = -1
    scrollToBottom()
}

const newConversation = () => {
    historyList.value.unshift({
        title: '新对话',
        updatedAt: Date.now(),
        messages: [{
            role: 'assistant',
            content: '你好！我是 Doctor.Song，你的专属医学AI助手。我学习了大量医学知识，可以回答你的健康问题。有什么我可以帮助你的吗？',
            thinkText: '',
            thinkOpen: true,
            thinkDone: true,
            streaming: false,
            thinkTime: null,
        }]
    })
    currentConversationIndex.value = 0
    editingIndex.value = -1
    scrollToBottom()
}

const toggleThink = (msgIndex) => {
    currentConversation.value.messages[msgIndex].thinkOpen =
        !currentConversation.value.messages[msgIndex].thinkOpen
}

const stopStreaming = () => {
    if (activeAbortCtrl) {
        activeAbortCtrl.abort()
        activeAbortCtrl = null
    }
}

// --- Conversation management ---
const getPreview = (item) => {
    const lastMsg = item.messages.slice(-1)[0]
    if (!lastMsg) return ''
    const text = lastMsg.content || ''
    return text.length > 30 ? text.slice(0, 30) + '...' : text
}

const formatTime = (ts) => {
    if (!ts) return ''
    const date = new Date(ts)
    const now = new Date()
    const diff = now - date
    if (diff < 60000) return '刚刚'
    if (diff < 3600000) return Math.floor(diff / 60000) + ' 分钟前'
    if (diff < 86400000) return Math.floor(diff / 3600000) + ' 小时前'
    return date.toLocaleDateString('zh-CN')
}

const startEdit = (index, title) => {
    editingIndex.value = index
    editTitle.value = title
    nextTick(() => {
        const inputs = document.querySelectorAll('.history-title-input')
        if (inputs.length > 0) inputs[0].focus()
    })
}

const finishEdit = (index) => {
    if (editTitle.value.trim()) {
        historyList.value[index].title = editTitle.value.trim()
    }
    editingIndex.value = -1
}

const deleteConversation = (index) => {
    if (historyList.value.length <= 1) {
        ElMessage.warning('至少保留一个对话')
        return
    }
    historyList.value.splice(index, 1)
    if (currentConversationIndex.value >= historyList.value.length) {
        currentConversationIndex.value = historyList.value.length - 1
    }
    if (currentConversationIndex.value === index) {
        currentConversationIndex.value = Math.min(index, historyList.value.length - 1)
    }
    editingIndex.value = -1
}

// --- Stream parsing ---
const parseStreamBuffer = (buffer) => {
    let thinkText = ''
    let content = ''
    let thinkDone = false

    const thinkStart = buffer.indexOf('<think>')
    const thinkEnd = buffer.indexOf('</think>')

    if (thinkStart !== -1 && thinkEnd !== -1) {
        thinkText = buffer.slice(thinkStart + 7, thinkEnd).trim()
        thinkDone = true
        let after = buffer.slice(thinkEnd + 8)
        after = after.replace('<answer>', '').replace('</answer>', '')
        content = after.trim()
    } else if (thinkStart !== -1 && thinkEnd === -1) {
        thinkText = buffer.slice(thinkStart + 7).trim()
        thinkDone = false
    } else {
        let text = buffer.replace('<answer>', '').replace('</answer>', '')
        content = text.trim()
        thinkDone = true
    }

    return { thinkText, content, thinkDone }
}

// --- Send message ---
const sendMessage = async () => {
    const text = userInput.value.trim()
    if (!text || isLoading.value) return

    currentConversation.value.messages.push({ role: 'user', content: text, thinkText: '', streaming: false })

    if (currentConversation.value.title === '新对话') {
        currentConversation.value.title = text.length > 20 ? text.slice(0, 20) + '...' : text
    }
    currentConversation.value.updatedAt = Date.now()

    userInput.value = ''
    scrollToBottom()

    const assistantMsg = {
        role: 'assistant',
        content: '',
        thinkText: '',
        thinkOpen: true,
        thinkDone: false,
        streaming: true,
        loading: true,
        thinkTime: null,
    }
    currentConversation.value.messages.push(assistantMsg)
    isLoading.value = true
    let thinkStartTime = null
    scrollToBottom()

    // Build messages for API
    const rawMessages = []
    for (const msg of currentConversation.value.messages) {
        if (msg.loading && !msg.content && !msg.thinkText) continue
        if (msg.streaming && !msg.content && !msg.thinkText) continue
        rawMessages.push({
            role: msg.role === 'assistant' ? 'assistant' : 'user',
            content: msg.content
        })
    }
    const firstUserIdx = rawMessages.findIndex(m => m.role === 'user')
    const settings = getSettings()
    const apiMessages = [
        { role: 'system', content: settings.systemPrompt },
        ...(firstUserIdx > 0 ? rawMessages.slice(firstUserIdx) : rawMessages)
    ]

    let fullBuffer = ''

    try {
        activeAbortCtrl = new AbortController()

        const response = await fetch(`${BASE_URL}${API.CHAT_COMPLETIONS}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'doctor-song',
                messages: apiMessages,
                stream: true,
                temperature: settings.temperature,
                max_tokens: settings.maxLength,
                enable_thinking: true,
            }),
            signal: activeAbortCtrl.signal,
        })

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let lineBuffer = ''

        while (true) {
            const { done, value } = await reader.read()
            if (done) break

            lineBuffer += decoder.decode(value, { stream: true })
            const lines = lineBuffer.split('\n')
            lineBuffer = lines.pop() || ''

            for (const line of lines) {
                const trimmed = line.trim()
                if (!trimmed || !trimmed.startsWith('data: ')) continue
                const data = trimmed.slice(6)
                if (data === '[DONE]') continue

                try {
                    const json = JSON.parse(data)
                    const delta = json.choices?.[0]?.delta || {}
                    // Handle llama.cpp reasoning_content (native template format)
                    const reasonToken = delta.reasoning_content
                    if (reasonToken) {
                        if (!thinkStartTime) thinkStartTime = Date.now()
                        assistantMsg.thinkText = (assistantMsg.thinkText || '') + reasonToken
                        assistantMsg.loading = false
                        scrollToBottom()
                    }
                    // Handle content token (both old <think> tag and direct content formats)
                    const contentToken = delta.content
                    if (contentToken) {
                        fullBuffer += contentToken
                        const parsed = parseStreamBuffer(fullBuffer)
                        // Only use tag-based parsing for thinkText if we haven't
                        // already received reasoning_content via the native format
                        if (!assistantMsg.thinkText && parsed.thinkText) {
                            if (!thinkStartTime) thinkStartTime = Date.now()
                            assistantMsg.thinkText = parsed.thinkText
                        }
                        assistantMsg.content = parsed.content
                        if (!assistantMsg.thinkDone) {
                            assistantMsg.thinkDone = parsed.thinkDone
                        }
                        if (parsed.thinkDone && thinkStartTime && assistantMsg.thinkTime == null) {
                            assistantMsg.thinkTime = ((Date.now() - thinkStartTime) / 1000).toFixed(1)
                        }
                        assistantMsg.loading = false
                        scrollToBottom()
                    }
                } catch (e) { /* skip */ }
            }
        }

        const finalParsed = parseStreamBuffer(fullBuffer)
        // Only use tag-parsed thinkText if we haven't accumulated any via reasoning_content
        if (!assistantMsg.thinkText && finalParsed.thinkText) {
            assistantMsg.thinkText = finalParsed.thinkText
        }
        assistantMsg.content = finalParsed.content || assistantMsg.content
        assistantMsg.thinkDone = true
        if (thinkStartTime && assistantMsg.thinkTime == null) {
            assistantMsg.thinkTime = ((Date.now() - thinkStartTime) / 1000).toFixed(1)
        }

    } catch (error) {
        if (error.name === 'AbortError') {
            if (!assistantMsg.content && !assistantMsg.thinkText) {
                assistantMsg.content = '(已停止生成)'
            }
        } else {
            console.error('Stream error:', error)
            ElMessage.error('获取回复失败，请确认API服务已启动')
            if (!assistantMsg.content && !assistantMsg.thinkText) {
                assistantMsg.content = '抱歉，获取回复失败，请稍后重试。'
            }
        }
    } finally {
        assistantMsg.streaming = false
        assistantMsg.loading = false
        assistantMsg.thinkDone = true
        isLoading.value = false
        activeAbortCtrl = null
        currentConversation.value.updatedAt = Date.now()
        scrollToBottom()
    }
}

onMounted(() => {
    scrollToBottom()
})
</script>

<style scoped>
/* === Layout === */
.chat-container {
    display: flex;
    height: 100vh;
    font-family: var(--font-sans);
    background: var(--bg-body);
    transition: background-color 0.3s ease;
}

/* === History panel === */
.history-panel {
    width: 280px;
    background: var(--bg-card);
    padding: 16px;
    overflow-y: auto;
    border-right: 1px solid var(--border-light);
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: background-color 0.3s ease, border-color 0.3s ease;
}

.new-chat-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 10px;
    background: var(--accent-gradient);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: opacity 0.2s ease, transform 0.15s ease;
    flex-shrink: 0;
}

.new-chat-btn:hover { opacity: 0.92; }
.new-chat-btn:active { transform: scale(0.98); }

.history-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    overflow-y: auto;
}

.history-item {
    display: flex;
    align-items: center;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: background-color 0.15s ease;
    border: 1px solid transparent;
    gap: 4px;
}

.history-item:hover {
    background: var(--bg-hover);
}

.history-item.active {
    background: var(--accent-light);
    border-color: rgba(59, 130, 246, 0.2);
}

.history-item-content {
    flex: 1;
    min-width: 0;
}

.history-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}

.history-title-input {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    background: var(--bg-input);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 2px 6px;
    width: 100%;
    outline: none;
    font-family: var(--font-sans);
}

.history-preview {
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-time {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
}

.history-actions {
    display: flex;
    gap: 2px;
    opacity: 0;
    transition: opacity 0.15s ease;
    flex-shrink: 0;
}

.history-item:hover .history-actions {
    opacity: 1;
}

.action-btn {
    width: 28px !important;
    height: 28px !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    color: var(--text-muted) !important;
}

.action-btn:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

.history-empty {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 13px;
}

/* === Chat wrapper === */
.chat-wrapper {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    background: var(--bg-body);
    transition: background-color 0.3s ease;
}

.chat-panel {
    width: 100%;
    max-width: 860px;
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* === Messages === */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 32px;
}

.message {
    display: flex;
    margin-bottom: 24px;
    gap: 12px;
}

.message .avatar {
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    flex-shrink: 0;
}

.message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: var(--radius-sm);
}

.message .bubble {
    max-width: 82%;
    font-size: 15px;
    line-height: 1.75;
    word-break: break-word;
    min-width: 0;
}

.message.user {
    flex-direction: row-reverse;
}

.message.user .bubble {
    background: var(--bg-bubble-user);
    border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
    padding: 12px 18px;
    color: var(--text-primary);
}

.message.assistant .bubble {
    padding: 4px 0;
    color: var(--text-primary);
}

/* === Think block === */
.think-block {
    background: var(--bg-think);
    border: 1px solid var(--border-think);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    margin-bottom: 12px;
}

.think-header {
    font-size: 13px;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
    font-weight: 500;
}

.think-header:hover {
    opacity: 0.8;
}

.think-time {
    font-weight: normal;
    color: var(--text-muted);
    font-size: 12px;
}

.think-content {
    margin-top: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    border-top: 1px solid var(--border-think);
    padding-top: 8px;
}

.think-spinner {
    width: 12px;
    height: 12px;
    border: 2px solid rgba(59, 130, 246, 0.2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* === Markdown content === */
.answer-text :deep(h1),
.answer-text :deep(h2),
.answer-text :deep(h3) {
    margin: 16px 0 8px;
    font-weight: 600;
    line-height: 1.4;
}

.answer-text :deep(h1) { font-size: 1.4em; }
.answer-text :deep(h2) { font-size: 1.2em; }
.answer-text :deep(h3) { font-size: 1.05em; }

.answer-text :deep(p) {
    margin: 8px 0;
}

.answer-text :deep(ul), .answer-text :deep(ol) {
    padding-left: 20px;
    margin: 8px 0;
}

.answer-text :deep(li) {
    margin: 4px 0;
}

.answer-text :deep(code) {
    background: var(--bg-hover);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.88em;
    font-family: var(--font-mono);
}

.answer-text :deep(pre) {
    background: var(--bg-sidebar);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    overflow-x: auto;
    margin: 12px 0;
    font-size: 13px;
    line-height: 1.6;
}

.answer-text :deep(pre code) {
    background: none;
    padding: 0;
}

.answer-text :deep(table) {
    border-collapse: collapse;
    margin: 12px 0;
    width: 100%;
}

.answer-text :deep(th), .answer-text :deep(td) {
    border: 1px solid var(--border-card);
    padding: 8px 12px;
    text-align: left;
    font-size: 14px;
}

.answer-text :deep(th) {
    background: var(--bg-hover);
    font-weight: 600;
}

.answer-text :deep(blockquote) {
    border-left: 3px solid var(--accent);
    padding-left: 14px;
    margin: 10px 0;
    color: var(--text-secondary);
}

.answer-text :deep(a) {
    color: var(--accent);
    text-decoration: underline;
}

.answer-text :deep(strong) {
    font-weight: 600;
}

.answer-text :deep(hr) {
    border: none;
    border-top: 1px solid var(--border-light);
    margin: 16px 0;
}

/* === Streaming cursor === */
.stream-cursor {
    color: var(--accent);
    animation: blink 1s step-end infinite;
    font-weight: bold;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

/* === Loading dots === */
.loading-dots {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 0;
}

.loading-dots span {
    width: 7px;
    height: 7px;
    background: var(--text-muted);
    border-radius: 50%;
    animation: dot-bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }
.loading-dots span:nth-child(3) { animation-delay: 0s; }

@keyframes dot-bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* === Input area === */
.input-area {
    padding: 16px 32px 20px;
}

.input-wrapper {
    position: relative;
    display: flex;
    align-items: center;
}

input {
    width: 100%;
    padding: 14px 52px 14px 20px;
    border: 1px solid var(--border-card);
    border-radius: var(--radius-full);
    font-size: 15px;
    background: var(--bg-card);
    color: var(--text-primary);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    font-family: var(--font-sans);
    outline: none;
}

input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

input::placeholder { color: var(--text-placeholder); }
input:disabled { opacity: 0.6; }

.button-group {
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    align-items: center;
}

.circle-btn {
    width: 38px;
    height: 38px;
    border: none;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.15s ease;
}

.send-btn {
    background: var(--accent-gradient);
    color: white;
}

.send-btn:disabled {
    background: var(--bg-hover);
    color: var(--text-muted);
    cursor: default;
}

.send-btn:not(:disabled):hover {
    opacity: 0.9;
}

.send-btn:not(:disabled):active {
    transform: scale(0.95);
}

.stop-btn {
    background: var(--danger);
    color: white;
}

.stop-btn:hover {
    background: var(--danger-hover);
}

/* === Welcome hint === */
.welcome-hint {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    gap: 8px;
}

.welcome-logo {
    width: 64px;
    height: 64px;
    border-radius: var(--radius-lg);
    margin-bottom: 8px;
    box-shadow: 0 2px 12px rgba(59, 130, 246, 0.1);
}

.welcome-hint h2 {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
}

.welcome-hint p {
    font-size: 14px;
    color: var(--text-secondary);
    max-width: 360px;
    line-height: 1.6;
}

.welcome-disclaimer {
    font-size: 12px !important;
    color: var(--text-muted) !important;
    margin-top: 12px;
}
</style>
