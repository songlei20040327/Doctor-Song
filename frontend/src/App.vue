<script setup>
import { RouterView, useRouter } from 'vue-router'
import { onMounted } from 'vue'

const router = useRouter()

onMounted(() => {
    if (router.currentRoute.value.name === 'empty') {
        router.push('/login')
    }
})
</script>

<template>
    <RouterView />
</template>

<style>
html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
}

#app {
    padding: 0;
    height: 100%;
    width: 100%;
}
</style>
