<template>
    <el-container class="layout-container" style="height: 100%">
        <!-- left sidebar -->
        <el-aside width="72px" class="sidebar">
            <el-scrollbar>
                <div class="nav-logo" @click="goToLogin">
                    <img src="@/assets/logo.png" alt="Logo"/>
                </div>

                <div class="nav-menu">
                    <div class="nav-menu-item"
                         :class="{ 'active': currentComponent === 'ChatPage' }"
                         @click="selectComponent('ChatPage')"
                         title="对话">
                        <el-icon :size="22"><ChatDotSquare/></el-icon>
                        <span>对话</span>
                    </div>
                    <div class="nav-menu-item"
                         :class="{ 'active': currentComponent === 'EmptyPage1' }"
                         @click="selectComponent('EmptyPage1')"
                         title="知识库">
                        <el-icon :size="22"><Document/></el-icon>
                        <span>知识库</span>
                    </div>
                    <div class="nav-menu-item"
                         :class="{ 'active': currentComponent === 'EmptyPage2' }"
                         @click="selectComponent('EmptyPage2')"
                         title="笔记">
                        <el-icon :size="22"><Edit/></el-icon>
                        <span>笔记</span>
                    </div>
                    <div class="nav-menu-item"
                         :class="{ 'active': currentComponent === 'EmptyPage3' }"
                         @click="selectComponent('EmptyPage3')"
                         title="我的">
                        <el-icon :size="22"><User/></el-icon>
                        <span>我的</span>
                    </div>
                </div>

                <div class="bottom-icons">
                    <div class="nav-menu-item" @click="toggleDark" title="切换主题">
                        <el-icon :size="18">
                            <Sunny v-if="isDark" />
                            <Moon v-else />
                        </el-icon>
                    </div>
                    <div class="nav-menu-item" @click="showSettings = true" title="设置">
                        <el-icon :size="18"><Setting/></el-icon>
                    </div>
                </div>
            </el-scrollbar>
        </el-aside>

        <!-- main content -->
        <el-main>
            <component :is="showComponent"></component>
        </el-main>
    </el-container>

    <!-- Settings Drawer -->
    <el-drawer
        v-model="showSettings"
        title="设置"
        direction="rtl"
        size="380px"
        :z-index="3000"
    >
        <SettingsPanel @close="showSettings = false" />
    </el-drawer>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ChatDotSquare, Document, Edit, User, Setting, Sunny, Moon } from '@element-plus/icons-vue'
import ChatPage from './chat/ChatPage.vue'
import EmptyPage from './empty/EmptyPage.vue'
import SettingsPanel from './SettingsPanel.vue'

const showComponent = ref(ChatPage)
const currentComponent = ref('ChatPage')
const showSettings = ref(false)
const router = useRouter()

const isDark = ref(false)

// init dark mode from localStorage or system preference
const initDark = () => {
    const stored = localStorage.getItem('doctor_song_dark_mode')
    if (stored !== null) {
        isDark.value = stored === 'true'
    } else {
        isDark.value = window.matchMedia('(prefers-color-scheme: dark)').matches
    }
    applyDark()
}
const applyDark = () => {
    document.documentElement.classList.toggle('dark', isDark.value)
    localStorage.setItem('doctor_song_dark_mode', isDark.value)
}
const toggleDark = () => {
    isDark.value = !isDark.value
    applyDark()
}

initDark()

const selectComponent = (component) => {
    currentComponent.value = component
    if (component === 'ChatPage') {
        showComponent.value = ChatPage
    } else {
        showComponent.value = EmptyPage
    }
}

const goToLogin = () => {
    router.push('/login')
}
</script>

<style scoped>
.sidebar {
    position: relative;
    overflow: hidden;
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border-light);
    display: flex;
    flex-direction: column;
    transition: background-color 0.3s ease;
}

.nav-logo {
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    padding: 12px 0;
}

.nav-logo img {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    transition: transform 0.2s ease;
}

.nav-logo img:hover {
    transform: scale(1.08);
}

.nav-menu {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 8px;
    gap: 4px;
}

.nav-menu-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
    position: relative;
}

.nav-menu-item span {
    font-size: 10px;
    margin-top: 3px;
    color: var(--text-muted);
}

.nav-menu-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
}

.nav-menu-item.active {
    color: var(--accent);
    background: var(--accent-light);
}

.nav-menu-item.active span {
    color: var(--accent);
}

.bottom-icons {
    position: absolute;
    bottom: 16px;
    left: 0;
    right: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
}

.layout-container .el-menu {
    border-right: none;
}

.layout-container .el-main {
    padding: 0;
    background: var(--bg-body);
    transition: background-color 0.3s ease;
}

.el-scrollbar {
    height: 100%;
}
</style>
