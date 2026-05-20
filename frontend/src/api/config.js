export const API = {
    CHAT_COMPLETIONS: '/v1/chat/completions',
    LIST_MODELS: '/v1/models',
}

// Dev: Vite proxy handles /v1 -> localhost:8000
// Prod: set VITE_API_BASE_URL env var or use same-origin deployment
export const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
