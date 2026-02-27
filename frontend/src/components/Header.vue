<template>
    <div class="title">
        <span class="hover-bold" @click="is_show_help = ! is_show_help">ModelViewer</span>
    </div>
    <div v-if="is_show_help" class="float-info-window">
        <!-- item -->
        <p>ModelViewer is based on LLM-Viewer to visualize AI models and analyze the deployment on hardware devices.</p>
        <p>At the center of the page, you can see the graph of the model. Click the node to see the detail of the node.</p>
        <p>
            ← At the left of the page, you can see the configuration pannel.
            You can select the backend server, the model, and hardware device.
            You can set the inference config and optimization config.
        </p>
        <p>Click "analyze" button to start the analysis.</p>
        <p>
            ↙ The Network-wise Analysis result is demonstrated in the left pannel.
        </p>
        <p>
            We invite you to read our paper <a class="hover-bold" href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank">LLM Inference Unveiled: Survey and Roofline Model Insights</a>.
            In this paper, we provide a comprehensive analysis of the latest advancements in efficient LLM inference using LLM-Viewer. 
            Citation bibtext:
        </p>
    </div>
</template>

<script setup>
import { ref } from 'vue';

const version = ref(llm_viewer_frontend_version)

const is_show_help = ref(false)
</script>

<style scoped>
.active {
    color: #fff;
    background-color: #5b5b5b;
}

.title {
    font-size: 24px;
    text-align: left;
    padding-left: 20px;
    padding-right: 20px;
}

.hover-bold {
    color: inherit;
    /* text-decoration: none; */
}

.hover-bold:hover {
    font-weight: bold;
}

.float-info-window {
    position: absolute;
    top: 80px;
    left: 40%;
    height: auto;
    width: 30%;
    background-color: #f1f1f1ed;
    padding: 20px;
    /* background-color: #fff; */
    /* border: 2px solid #4e4e4e; */
    z-index: 999;
}
</style>