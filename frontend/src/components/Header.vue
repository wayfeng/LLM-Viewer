<template>
    <div class="title">
        <a href="https://github.com/wayfeng/LLM-Viewer" target="_blank" class="hover-bold">LLM-Viewer</a>
        v{{ version }}
    </div>
    <div>
        <span> | </span>
        <a href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank" class="hover-bold">Paper</a>
    </div>
    <div>
        <span> | </span>
        <span class="hover-bold" @click="is_show_help = ! is_show_help">Help</span>
    </div>
    <div v-if="is_show_help" class="float-info-window">
        <!-- item -->
        <p>LLM-Viewer is a open-sourced tool to visualize the LLM model and analyze the deployment on hardware devices.</p>
        <p>
            At the center of the page, you can see the graph of the LLM model. Click the node to see the detail of the node.
        </p>
        <p>↑ At the top of the page, you can set the LLM model, hardware devices, and server.
            If you deploy the LLM-Viewer localhost, you can select the localhost server.
        </p>
        <p>
            ← At the left of the page, you can see the configuration pannel. You can set the inference config and optimization config.
        </p>
        <p>
            ↙ The Network-wise Analysis result is demonstrated in the left pannel.
        </p>
        <p>
            We invite you to read our paper <a class="hover-bold" href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank">LLM Inference Unveiled: Survey and Roofline Model Insights</a>.
            In this paper, we provide a comprehensive analysis of the latest advancements in efficient LLM inference using LLM-Viewer. 
            Citation bibtext:
        </p>
        @article{yuan2024llm,<br/>
            &nbsp    title={LLM Inference Unveiled: Survey and Roofline Model Insights},<br/>
            &nbsp    author={Yuan, Zhihang and Shang, Yuzhang and Zhou, Yang and Dong, Zhen and Xue, Chenhao and Wu, Bingzhe and Li, Zhikai and Gu, Qingyi and Lee, Yong Jae and Yan, Yan and others},<br/>
            &nbsp    journal={arXiv preprint arXiv:2402.16363},<br/>
            &nbsp    year={2024}<br/>
        }
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
    font-size: 18px;
    text-align: left;
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