<template>
    <div class="login-page">
        <div class="content">
            <div class="center-content">
                <div class="logo-area">
                    <img src="@/assets/logo.png" alt="Doctor.Song" class="logo" />
                </div>
                <h1>Doctor.Song</h1>
                <p class="tagline">专业医学AI助手，为您提供可靠的医疗健康咨询</p>
                <div class="features">
                    <span class="feature">循证医学知识</span>
                    <span class="dot">·</span>
                    <span class="feature">流式实时响应</span>
                    <span class="dot">·</span>
                    <span class="feature">隐私安全保护</span>
                </div>
                <button class="cta-button" @click="goToMainPage">
                    开始对话
                </button>
                <p class="disclaimer">AI 生成的医学建议仅供参考，不能替代专业医生诊断</p>
            </div>
        </div>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const goToMainPage = () => {
    router.push('/main')
}
</script>

<style scoped>
.login-page {
    position: relative;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(160deg, #f8f6f3 0%, #e8e6e1 40%, #eef2f9 100%);
    overflow: hidden;
}

.login-page::before {
    content: '';
    position: absolute;
    top: -200px;
    right: -200px;
    width: 600px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.06) 0%, transparent 70%);
    pointer-events: none;
}

.login-page::after {
    content: '';
    position: absolute;
    bottom: -150px;
    left: -150px;
    width: 500px;
    height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.04) 0%, transparent 70%);
    pointer-events: none;
}

.content {
    position: relative;
    z-index: 1;
}

.center-content {
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
}

.logo-area {
    margin-bottom: 8px;
}

.logo {
    width: 72px;
    height: 72px;
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(59, 130, 246, 0.15);
}

h1 {
    font-size: 2.4em;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.tagline {
    font-size: 1.05em;
    color: var(--text-secondary);
    max-width: 360px;
    line-height: 1.6;
    margin: 0;
}

.features {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88em;
    color: var(--text-muted);
    margin-top: 4px;
}

.dot {
    color: var(--border-card);
}

.cta-button {
    margin-top: 24px;
    padding: 14px 48px;
    font-size: 1.05em;
    font-weight: 600;
    color: #fff;
    background: var(--accent-gradient);
    border: none;
    border-radius: var(--radius-full);
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 2px 12px rgba(59, 130, 246, 0.25);
    letter-spacing: 0.5px;
}

.cta-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.35);
}

.cta-button:active {
    transform: translateY(0);
}

.disclaimer {
    margin-top: 28px;
    font-size: 0.78em;
    color: var(--text-muted);
    max-width: 340px;
    line-height: 1.5;
}
</style>
