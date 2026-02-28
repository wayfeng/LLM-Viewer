<script setup>

import Graph from "./components/Graph.vue"
import LeftPannel from "./components/LeftPannel.vue"
import Header from "./components/Header.vue"

import { ref, provide } from 'vue';

const model_id = ref(import.meta.env.VITE_MODEL_ID);
const hardware = ref(import.meta.env.VITE_HARDWARE);
const global_update_trigger = ref(1);
const total_results = ref({});
const ip_port = ref(import.meta.env.VITE_IP_PORT);

provide("model_id", model_id);
provide("hardware", hardware);
provide("global_update_trigger", global_update_trigger);
provide("total_results", total_results);
provide("ip_port", ip_port);


const global_inference_config = ref({ 
  stage: "decode",
  batchsize: 1,
  seqlen: 1024,
  genlen: 1024,
  tp_size: 1,
  w_bit: 8,
  a_bit: 8,
  kv_bit: 8,
  use_flashattention: false,
  fp16_tops: 0,
  int8_tops: 0,
  bandwidth: 0,
  onchip_buffer: 0,
});
provide("global_inference_config", global_inference_config);

</script>

<template>
  <div class="app_container">
    <div class="upper_header">
      <Header></Header>
    </div>
    <div class="bottom-block">
      <LeftPannel></LeftPannel>
      <Graph></Graph>
    </div>

  </div>
</template>

<style>
body {
  overflow-x: hidden;
  overflow-y: hidden;
}

.app_container {
  /* display: flex;
  flex-direction: column;
  width: 98vw; */
  width: 100%;
  height: 100vh;
}

.upper_header {
  flex: 1;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 50px;
  background-color: #f0f0f0;
  /* border-right: 1px solid #e2e2e2; */
  border-bottom: 3px solid #e2e2e2;
}

.bottom-block {
  display: flex;
  flex-direction: row;
  height: calc(100% - 60px);
}
</style>
