<template>
    <h2>Configuration</h2>
    <div class="config_div">
        Server:
        <select v-model="ip_port">
            <option value="172.16.112.118:5000">172.16.112.118</option>
            <option value="172.16.112.60:5000">172.16.112.60</option>
            <option value="172.16.112.46:5000">172.16.112.46</option>
            <option value="127.0.0.1:5000">localhost</option>
        </select>
    </div>
    <div class="config_div">
        Model:
        <select v-model="select_model_id">
            <option v-for="model_id in available_model_ids" :value="model_id">{{ model_id }}</option>
        </select>
    </div>
    <h3>Hardware Config</h3>
    <div class="config_div">
        <span>Hardware: </span>
        <select v-model="select_hardware">
            <option v-for="hardware in available_hardwares" :value="hardware">{{ hardware }}</option>
        </select>
    </div>
    <div class="config_div">
        FP16
        <input type="number" v-model="fp16_tops" min="1" step="0.1"> TOPS
    </div>
    <div class="config_div">
        INT8
        <input type="number" v-model="int8_tops" min="1" step="0.1"> TOPS
    </div>
    <div class="config_div">
        Memory Bandwidth:
        <input type="number" v-model="memory_bandwidth" min="1" step="0.1"> GB/s
    </div>
    <div class="config_div">
        On-chip Buffer:
        <input type="number" v-model="onchip_cache" min="1" step="0.1"> MB
    </div>
    <h3>Inference Config</h3>
    <div class="config_div">
        Stage:
        <input type="radio" v-model="inference_stage" id="decode" value="decode" checked>
        <label for="decode">Decode</label>
        <input type="radio" v-model="inference_stage" id="prefill" value="prefill">
        <label for="prefill">Prefill</label>
        <input type="radio" v-model="inference_stage" id="chat" value="chat">
        <label for="prefill">Chat</label>
    </div>
    <div class="config_div">
        Batch size
        <input type="number" v-model.lazy="batch_size" min="1" max="256">
    </div>
    <!-- <div class="config_div" v-if="inference_stage!=chat"> -->
    <div class="config_div" v-if="inference_stage!='chat'">
        Prompt Length
        <input type="number" v-model.lazy="seq_length" min="1" max="4096">
    </div>
    <div class="config_div" v-else>
        Prompt Length
        <input type="number" v-model.lazy="seq_length" min="1" max="4096">
        <br/>
        Generate Length
        <!-- <span id="seq_length">1024</span> -->
        <input type="number" v-model.lazy="gen_length" min="1" max="4096">
    </div>
    <div class="config_div">
        Tensor parallelism
        <select v-model="tp_size">
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="4">4</option>
            <option value="8">8</option>
        </select>
    </div>
    <h3>Optimization Config</h3>
    <div class="config_div">
        Weight Quantization
        <select v-model="w_quant">
            <option value="16">FP16</option>
            <option value="8">8-bit</option>
            <option value="4">4-bit</option>
            <option value="2">2-bit</option>
            <option value="1">1-bit</option>
        </select>
    </div>
    <div class="config_div">
        Activation Quantization
        <select v-model="a_quant">
            <option value="16">FP16</option>
            <option value="8">8-bit</option>
            <option value="4">4-bit</option>
            <option value="2">2-bit</option>
            <option value="1">1-bit</option>
        </select>
    </div>
    <div class="config_div">
        KV Cache Quantization
        <select v-model="kv_quant">
            <option value="16">FP16</option>
            <option value="8">8-bit</option>
            <option value="4">4-bit</option>
            <option value="2">2-bit</option>
            <option value="1">1-bit</option>
        </select>
    </div>
    <div class="config_div">
        Use FlashAttention
        <input type="checkbox" v-model="use_flashattention">
    </div>
    <div class="button_div">
        <button type="button" @click="trigger_analyze" style="width: 70%;">Analyze</button>
    </div>
    <h2>Network-wise Analysis</h2>
    <div>
        <h3>{{ inference_stage }}</h3>
        <div v-for="(value, key) in total_results[inference_stage]" :key="key" class="network-wise-info-item">
            <span v-if="['bound'].includes(key)" class="highlight-span">{{ key }}: {{ value }}</span>
            <span v-else-if="['inference_time'].includes(key)" class="highlight-time">{{ key }}: {{ strNumberTime(value) }}</span>
            <span v-else-if="['OPs'].includes(key)" class="highlight-ops">{{ key }}: {{ strNumber(value, 'OPs') }}</span>
            <span v-else>{{ key }}: {{ strNumber(value, 'B') }}</span>
        </div>
        <p>NOTE: The time estimated by the roofline model represents the theoretical performance that the hardware can achieve. 
        The purpose of creating this tool is to help readers gain a clearer understanding of the key factors that influence LLM inference. 
        Only the relative relationships can be referenced. </p>
    </div>
</template>

<script setup>
import { inject, ref, watch, onMounted } from 'vue';
import { strNumber, strNumberTime } from '@/utils.js';
import axios from 'axios'

const global_update_trigger = inject('global_update_trigger');

const global_inference_config = inject('global_inference_config');
const total_results = inject('total_results');

const inference_stage = ref('decode');
const batch_size = ref(1);
const seq_length = ref(1024);
const gen_length = ref(1024);
const tp_size = ref(1);
const w_quant = ref(8);
const a_quant = ref(8);
const kv_quant = ref(8);
const use_flashattention = ref(false);
const fp16_tops = ref(450);
const int8_tops = ref(900);
const memory_bandwidth = ref(1536);
const onchip_cache = ref(24);
const max_ops = ref(0);

function trigger_analyze() {
    global_inference_config.value.stage = inference_stage.value
    global_inference_config.value.batchsize = batch_size.value
    global_inference_config.value.seqlen = seq_length.value
    global_inference_config.value.genlen = gen_length.value
    global_inference_config.value.tp_size = tp_size.value
    global_inference_config.value.w_bit = w_quant.value
    global_inference_config.value.a_bit = a_quant.value
    global_inference_config.value.kv_bit = kv_quant.value
    global_inference_config.value.use_flashattention = use_flashattention.value
    global_inference_config.value.bandwidth = memory_bandwidth.value * 1e9
    global_inference_config.value.onchip_buffer = onchip_cache.value * 1e6
    global_inference_config.value.fp16_tops = fp16_tops.value * 1e12
    global_inference_config.value.int8_tops = int8_tops.value * 1e12
    global_update_trigger.value += 1
}

const model_id = inject('model_id');
const ip_port = inject('ip_port');
const hardware = inject('hardware');

const available_hardwares = ref([]);
const available_model_ids = ref([]);

function update_hardware_info() {
    const url = 'http://' + ip_port.value + '/get_hardware_params'
    axios.post(url, { hardware: hardware.value }).then(function (response) {
        console.log(response);
        fp16_tops.value = response.data.FP16
        int8_tops.value = response.data.INT8
        memory_bandwidth.value = response.data.bandwidth
        onchip_cache.value = response.data.onchip_buffer
    }).catch(function (error) {
        console.log("error in get_hardware_params");
        console.log(error);
    });
}

function update_available() {
    const url = 'http://' + ip_port.value + '/get_available'
    axios.get(url).then(function (response) {
        console.log(response);
        available_hardwares.value = response.data.available_hardwares
        available_model_ids.value = response.data.available_model_ids
    }).catch(function (error) {
        console.log("error in get_available");
        console.log(error);
    });
}

watch(ip_port, (n) => {
    console.log("ip_port", n)
    update_available()
})

var select_model_id = ref(model_id.value);
watch(select_model_id, (n) => {
    console.log("select_model_id", n)
    model_id.value = n
})

var select_hardware = ref(hardware.value);
watch(select_hardware, (n) => {
    console.log("select_hardware", n)
    hardware.value = n
    if (n != "custom") {
        //global_update_trigger.value += 1
        update_hardware_info()
    }
})

onMounted(() => {
    console.log("Left panel mounted")
    update_available()
})

</script>

<style>

.config_div{
    padding: 2px 0;
}

.button_div{
    display: flex;
    justify-content: center;
    margin-top: 20px;
    height: 50px;
}

.hover_color {
    color: #0000ff;
    cursor: pointer;
}

.network-wise-info-item {
    padding: 3px;
}

.highlight-span {
    color: #d7263d;
    font-weight: bold;
    background: #fffbe6;
    padding: 2px 6px;
    border-radius: 4px;
}

.highlight-time {
    color: #1e88e5;
}

.highlight-ops {
    color: #43a047;
}

</style>