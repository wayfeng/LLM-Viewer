import logging
import importlib
import math
from roofline_model import roofline_analyze
from model_params import load_model_params
from utils import print_params
from functools import lru_cache

logger = logging.getLogger(__name__)

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]

MODEL_ANALYZER_REGISTRY = {
    "LLMAnalyzer": ["qwen3", "qwen2", "qwen2_5", "llama", "chatglm"],
    "MoEAnalyzer": ["qwen3_moe", "qwen2_moe", "qwen2_5_moe", "gpt_oss"],
    "VLMAnalyzer": ["qwen3_vl", "qwen2_vl", "qwen2_5_vl"],
}


class ModelAnalyzer:
    def __init__(self, model_id, model_params=None):
        self.model_id = model_id
        self.model_params = model_params if model_params is not None else load_model_params(model_id)
        model_type = self.model_params["model_type"]
        self.module = importlib.import_module(f"models.{model_type.lower()}")
 
        # temporary variables for analysis
        self.results = None

    def analyze(self, **kwargs):
        """
        seqlen: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        pass

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1,
        bandwidth=None,
        fp16_tops=None,
        int8_tops=None,
        onchip_buffer=None
    ):
        configs = {
            "seqlen": prompt_len,
            "batchsize": batchsize,
            "w_bit": w_bit,
            "a_bit": a_bit,
            "kv_bit": kv_bit if kv_bit is not None else a_bit,
            "use_flashattention": use_flashattention,
            "tp_size": tp_size,
            "bandwidth": bandwidth,
            "fp16_tops": fp16_tops,
            "int8_tops": int8_tops,
            "onchip_buffer": onchip_buffer,
        }
        prefill_result = self.analyze(**configs)
        prefill_time = inference_time = prefill_result["total_results"]["prefill"]["inference_time"]
        inference_time += prefill_result["total_results"]["decode"]["inference_time"] * gen_len 

        return {"inference_time": inference_time * 1000, "prefill_time": prefill_time * 1000}

    def if_group_qa(self):
        """
        Returns whether the model uses Grouped Query Attention (GQA).
        """
        return (
            self.module.get_num_attention_heads(self.model_params) !=
            self.module.get_num_key_value_heads(self.model_params)
        )

    @print_params
    def _analyze_to_results(self, stage, name, OPs=0, load_weight=0, load_act=0,
        store_act=0, load_kv_cache=0, store_kv_cache=0, max_ops=0, bandwidth=0):
        memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_ops, OPs, memory_access)
        inference_time = OPs / performance
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }


class LLMAnalyzer(ModelAnalyzer):
    def __init__(self, model_id, model_params=None):
        super().__init__(model_id, model_params=model_params)

    def analyze_decode_layer(self, mode, **kwargs):
        tp_size = kwargs.get("tp_size", 1)
        seqlen = kwargs.get("seqlen", 1)
        batchsize = kwargs.get("batchsize", 1)
        w_bit = kwargs.get("w_bit", 16)
        a_bit = kwargs.get("a_bit", 16)
        kv_bit = kwargs.get("kv_bit", 16)
        fp16_tops = kwargs.get("fp16_tops")
        int8_tops = kwargs.get("int8_tops")
        bandwidth = kwargs.get("bandwidth")
        onchip_buffer = kwargs.get("onchip_buffer")
        use_flashattention = kwargs.get("use_flashattention", False)

        max_ops = int8_tops if w_bit <= 8 and a_bit <= 8 and kv_bit <= 8 else fp16_tops
        n = 1 if mode == "decode" else kwargs.get("seqlen", 1)

        w_byte = math.ceil(w_bit / 8)
        a_byte = math.ceil(a_bit / 8)
        kv_byte = math.ceil(kv_bit / 8)

        for name, (ic, oc) in self.module.get_linear_layers(self.model_params).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj

            self._analyze_to_results(
                mode,
                name,
                OPs=ic * oc * batchsize * n * 2 // tp_size,
                load_weight=ic * oc * w_byte // tp_size,
                load_act=ic * batchsize * n * a_byte // tp_size,
                store_act=0 if is_kv_proj else oc * batchsize * n * a_byte // tp_size,
                load_kv_cache=0,
                store_kv_cache=0 if is_normal_proj else oc * batchsize * n * kv_byte // tp_size,
                max_ops=max_ops,
                bandwidth=bandwidth
            )

        hidden_size = self.module.get_hidden_size(self.model_params)
        num_attention_heads = self.module.get_num_attention_heads(self.model_params)
        num_key_value_heads = self.module.get_num_key_value_heads(self.model_params)
        # for attention
        head_size = hidden_size // num_attention_heads
        # for decode
        qk_matmul_OPs = n * seqlen * head_size * num_attention_heads * batchsize * 2 // tp_size
        sv_matmul_OPs = n * head_size * seqlen * num_attention_heads * batchsize * 2 // tp_size
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * num_attention_heads * n * 1 * 5 // tp_size
        if use_flashattention:
            name = f"fused_attention"
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(n / block_size_r)
            q_numel = n * head_size * batchsize * num_attention_heads * a_byte // tp_size
            o_numel = n * seqlen * batchsize * num_attention_heads * a_byte // tp_size
            self._analyze_to_results(
                mode,
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2, # initialize 0 and save 0
                load_kv_cache=n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2 // tp_size,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                mode,
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=n * head_size * batchsize * num_attention_heads * a_byte // tp_size,
                store_act=n * seqlen * batchsize * num_attention_heads * a_byte // tp_size,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte // tp_size,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                mode,
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=n * seqlen * batchsize * num_attention_heads * a_byte // tp_size,
                store_act=n * head_size * batchsize * num_attention_heads * a_byte // tp_size,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte // tp_size,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                mode,
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * n * a_byte // tp_size,
                store_act=batchsize * num_attention_heads * seqlen * n * a_byte // tp_size,
                load_kv_cache=0,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )

        for name in self.module.get_norm_layers(self.model_params):
            # sum sub pow sum div mul add
            if "rmsnorm" in name:
                norm_OPs = batchsize * hidden_size * n * 4
            else:
                norm_OPs = batchsize * hidden_size * n * 7
            self._analyze_to_results(
                mode,
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * hidden_size * n * a_byte,
                store_act=batchsize * hidden_size * n * a_byte,
                load_kv_cache=0,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                mode,
                name,
                OPs=batchsize * hidden_size * n,
                load_weight=0,
                load_act=batchsize * hidden_size * n * a_byte // tp_size,
                store_act=batchsize * hidden_size * n * a_byte // tp_size,
                load_kv_cache=0,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )
        self._analyze_to_results(
            mode,
            "mlp_act",
            OPs=batchsize * hidden_size * n * 5 // tp_size,
            load_weight=0,
            load_act=batchsize * hidden_size * n * a_byte // tp_size,
            store_act=batchsize * hidden_size * n * a_byte // tp_size,
            load_kv_cache=0,
            store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
        )

    def analyze_chat(self, genlen, use_flashattention=False, **kwargs):
        total_results = self.results["total_results"]

        stage_results = self.results["prefill"]
        # seq_length:seq_length+gen_length
        total_results["chat"] = total_results["prefill"]
        for k, v in self.results["total_results"]["decode"].items():
            total_results["chat"][k] += v * genlen

        if use_flashattention:
            layer_graph = self.module.flashattention_transformer_layer_graph
        else:
            layer_graph = self.module.transformer_layer_graph

        for name, _ in layer_graph.items():
            if name in self.results["decode"]:
                stage_results[name]["OPs"] += self.results["decode"][name]["OPs"] * genlen
                stage_results[name]["memory_access"] += self.results["decode"][name]["memory_access"] * genlen

    def analyze_lm_head(self, total_results, **kwargs):
        batchsize = kwargs.get("batchsize", 1)
        bandwidth = kwargs.get("bandwidth")
        seqlen = kwargs.get("seqlen", 1)
        w_bit = kwargs.get("w_bit", 16)
        a_bit = kwargs.get("a_bit", 16)
        kv_bit = kwargs.get("kv_bit", a_bit)
        fp16_tops = kwargs.get("fp16_tops")
        int8_tops = kwargs.get("int8_tops")
        vocab_size = self.module.get_vocab_size(self.model_params)


        w_byte = math.ceil(w_bit / 8)
        a_byte = math.ceil(a_bit / 8)

        max_ops = int8_tops if w_bit <= 8 and a_bit <= 8 and kv_bit <= 8 else fp16_tops

        hidden_size = self.module.get_hidden_size(self.model_params)

        name = "lm_head"
        self._analyze_to_results(
            "decode", name,
            OPs=1 * batchsize * hidden_size * vocab_size * 2,
            load_weight=hidden_size * vocab_size * w_byte,
            load_act=1 * batchsize * hidden_size * a_byte,
            store_act=1 * batchsize * vocab_size * a_byte,
            load_kv_cache=0, store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
        )
        self._analyze_to_results(
            "prefill", name,
            OPs=seqlen * batchsize * hidden_size * vocab_size * 2,
            load_weight=1 * hidden_size * vocab_size * w_byte,
            load_act=seqlen * batchsize * hidden_size * a_byte,
            store_act=seqlen * batchsize * vocab_size * a_byte,
            load_kv_cache=0, store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
        )
        for dname in ALL_DATA_NAMES:
            total_results["decode"][dname] += self.results["decode"][name][dname]
            total_results["prefill"][dname] += self.results["prefill"][name][dname]

        return total_results

    def compute_total_results(self, **kwargs):
        num_hidden_layers = self.module.get_num_hidden_layers(self.model_params)
        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for _, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # memory footprint
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = 0
        for _, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = 0
        for _, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        return total_results

    def analyze(self, **kwargs):
        self.results = {"decode": {}, "prefill": {}}

        self.analyze_decode_layer("prefill", **kwargs)
        self.analyze_decode_layer("decode", **kwargs)

        # compute total
        total_results = self.compute_total_results(**kwargs)
        # lm_head
        self.analyze_lm_head(total_results, **kwargs)
        self.results["total_results"] = total_results
        return self.results


class MoEAnalyzer(LLMAnalyzer):
    def __init__(self, model_id, model_params=None):
        super().__init__(model_id, model_params=model_params)

    def analyze_decode_layer(self, mode, **kwargs):
        seqlen = kwargs.get("seqlen", 1)
        batchsize = kwargs.get("batchsize", 1)
        w_bit = kwargs.get("w_bit", 16)
        a_bit = kwargs.get("a_bit", 16)
        kv_bit = kwargs.get("kv_bit", a_bit)
        tp_size = kwargs.get("tp_size", 1)
        bandwidth = kwargs.get("bandwidth")
        fp16_tops = kwargs.get("fp16_tops")
        int8_tops = kwargs.get("int8_tops")
        onchip_buffer = kwargs.get("onchip_buffer")
        use_flashattention = kwargs.get("use_flashattention", False)

        max_ops = int8_tops if w_bit <= 8 and a_bit <= 8 and kv_bit <= 8 else fp16_tops
        n = 1 if mode == "decode" else seqlen

        w_byte = math.ceil(w_bit / 8)
        a_byte = math.ceil(a_bit / 8)
        kv_byte = math.ceil(kv_bit / 8)

        model_params = self.model_params
        num_attention_heads = self.module.get_num_attention_heads(model_params)
        hidden_size = self.module.get_hidden_size(model_params)
        num_key_value_heads = self.module.get_num_key_value_heads(model_params)
        num_experts = self.module.get_num_experts(model_params)
        num_active_experts = self.module.get_num_active_experts(model_params)
        moe_intermediate_size = self.module.get_moe_intermediate_size(model_params)

        for name, (ic, oc) in self.module.get_linear_layers(model_params).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            is_expert_proj = name in ["gate_proj", "up_proj", "down_proj"]

            if is_expert_proj:
                self._analyze_to_results(
                    mode,
                    name,
                    OPs=n * batchsize * ic * oc * 2 * num_active_experts // tp_size,
                    load_weight=1 * ic * oc * w_byte * min(num_active_experts * (batchsize + 1) // 2, num_experts) // tp_size,
                    load_act=n * batchsize * ic * a_byte * num_active_experts // tp_size,
                    store_act=n * batchsize * oc * a_byte * num_active_experts // tp_size,
                    load_kv_cache=0,
                    store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth,
                )
            else:
                self._analyze_to_results(
                    mode,
                    name,
                    OPs=n * ic * oc * batchsize * 2 // tp_size,
                    load_weight=1 * ic * oc * w_byte // tp_size,
                    load_act=n * ic * batchsize * a_byte // tp_size,
                    store_act=0 if is_kv_proj else n * oc * batchsize * a_byte // tp_size,
                    load_kv_cache=0,
                    store_kv_cache=0 if is_normal_proj else n * oc * batchsize * kv_byte // tp_size,
                    max_ops=max_ops, bandwidth=bandwidth
                )

        # for attention
        head_size = hidden_size // num_attention_heads

        qk_matmul_OPs = n * seqlen * head_size * num_attention_heads * batchsize * 2 // tp_size
        sv_matmul_OPs = n * head_size * seqlen * num_attention_heads * batchsize * 2 // tp_size
        softmax_OPs = batchsize * num_attention_heads * n * 5 // tp_size
        if use_flashattention:
            name = f"fused_attention"
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(n / block_size_r)
            q_numel = n * head_size * batchsize * num_attention_heads * a_byte
            o_numel = n * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                mode,
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel // tp_size,
                store_act=o_numel * 2 // tp_size,  # initialize O and save O
                load_kv_cache=n_blocks_r * seqlen * head_size * batchsize * num_key_value_heads * kv_byte * 2 // tp_size,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                mode,
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=n * head_size * batchsize * num_attention_heads * a_byte // tp_size,
                store_act=n * seqlen * batchsize * num_attention_heads * a_byte // tp_size,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte // tp_size,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                mode,
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=n * seqlen * batchsize * num_attention_heads * a_byte // tp_size,
                store_act=n * head_size * batchsize * num_attention_heads * a_byte // tp_size,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte // tp_size,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                mode,
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=n * batchsize * num_attention_heads * seqlen * a_byte // tp_size,
                store_act=n * batchsize * num_attention_heads * seqlen * a_byte // tp_size,
                load_kv_cache=0,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )

        for name in self.module.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            if "rmsnorm" in name:
                norm_OPs = batchsize * hidden_size * n * 4
            else:
                norm_OPs = batchsize * hidden_size * n * 7

            self._analyze_to_results(
                mode,
                name,
                OPs=norm_OPs,
                load_weight=0,
                load_act=batchsize * hidden_size * n * a_byte,
                store_act=batchsize * hidden_size * n * a_byte,
                load_kv_cache=0,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )

        self._analyze_to_results(
            mode,
            "gate",
            OPs=n * batchsize * hidden_size * num_experts * 2,
            load_weight=n * batchsize * num_experts * hidden_size * w_byte,
            load_act=n * batchsize * num_experts * a_byte,
            store_act=n * batchsize * num_experts * a_byte,
            load_kv_cache=0,
            store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
        )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                mode,
                name,
                OPs=n * batchsize * hidden_size * 1 // tp_size,
                load_weight=0,
                load_act=n * batchsize * hidden_size * a_byte // tp_size,
                store_act=n * batchsize * hidden_size * a_byte // tp_size,
                load_kv_cache=0,
                store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
            )
        self._analyze_to_results(
            mode,
            "mlp_act",
            OPs=n * batchsize * hidden_size * 5 * num_active_experts // tp_size,
            load_weight=0,
            load_act=n * batchsize * hidden_size * a_byte * num_active_experts // tp_size,
            store_act=n * batchsize * hidden_size * a_byte * num_active_experts // tp_size,
            load_kv_cache=0,
            store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
        )
        self._analyze_to_results(
            mode,
            "mlp_matmul",
            OPs=n * batchsize * moe_intermediate_size * hidden_size * num_active_experts // tp_size,
            load_weight=0,
            load_act=n * batchsize * moe_intermediate_size * a_byte * num_active_experts // tp_size,
            store_act=n * batchsize * moe_intermediate_size * a_byte * num_active_experts // tp_size,
            load_kv_cache=0,
            store_kv_cache=0, max_ops=max_ops, bandwidth=bandwidth
        )

    def analyze(self, **kwargs):
        self.results = {"decode": {}, "prefill": {}}

        self.analyze_decode_layer("prefill", **kwargs)
        self.analyze_decode_layer("decode", **kwargs)

        total_results = self.compute_total_results(**kwargs)
        self.analyze_lm_head(total_results, **kwargs)

        self.results["total_results"] = total_results
        return self.results

class VLMAnalyzer(ModelAnalyzer):
    def __init__(self, model_id, model_params=None):
        super().__init__(model_id, model_params=model_params)

    def analyze(self, **kwargs):
        return super().analyze(**kwargs)

class YOLOAnalyzer(ModelAnalyzer):
    def __init__(self, model_id, model_params=None):
        super().__init__(model_id, model_params=model_params)

    def analyze(self, **kwargs):
        return super().analyze(**kwargs)


@lru_cache(maxsize=128)
def get_analyzer(model_id) -> ModelAnalyzer:
    params = load_model_params(model_id)
    model_type = params["model_type"]
    analyzer_class = None
    for class_name, types in MODEL_ANALYZER_REGISTRY.items():
        if model_type in types:
            analyzer_class = globals()[class_name]
            break
    if analyzer_class is None:
        raise ValueError(f"Unknown model_type: {model_type}")
    return analyzer_class(model_id, model_params=params)
