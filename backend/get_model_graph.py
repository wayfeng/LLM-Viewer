from model_analyzer import get_analyzer
from utils import str_number

def get_model_graph(model_id, inference_config):
    w_bit = int(inference_config["w_bit"])
    a_bit = int(inference_config["a_bit"])
    kv_bit = int(inference_config["kv_bit"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["genlen"])
    fp16_tops = float(inference_config["fp16_tops"])
    int8_tops = float(inference_config["int8_tops"])
    bandwidth = float(inference_config["bandwidth"])
    onchip_buffer = float(inference_config["onchip_buffer"])

    configs = {}
    for key in inference_config.keys():
        if key not in ["stage", "genlen"]:
            configs[key] = inference_config[key]

    analyzer = get_analyzer(model_id)
    result = analyzer.analyze(**configs)

    GQA = analyzer.if_group_qa()

    #TODO: hardware_info should be in frontend already
    hardware_info = {
        "bandwidth": bandwidth,
        "max_OPS": int8_tops if w_bit <= 8 and a_bit <= 8 and kv_bit <= 8 else fp16_tops,
        "onchip_buffer": onchip_buffer,
    }

    nodes = [{"label": "input", "id": "input",}]
    edges = []

    def write_to_node(name, OPs, memory_access, info, input_names=[]):
        node = {
            "label": name,
            "id": name,
            "description": f"OPs:{str_number(OPs)}, Access:{str_number(memory_access, 'B')}",
            "info": info,
        }
        if GQA and name in ["qk_matmul", "sv_matmul"]:
            node["label"] += "(GQA)"
        nodes.append(node)
        for input_name in input_names:
            edge = {"source": input_name, "target": name}
            edges.append(edge)

    if use_flashattention:
        layer_graph = analyzer.module.flashattention_transformer_layer_graph
    else:
        layer_graph = analyzer.module.transformer_layer_graph
    stage = inference_config["stage"]
    total_results = result["total_results"]
    stage_results = result["prefill"] if stage == "chat" else result[stage]

    #TODO: move to analyzer
    if stage == "chat":
        # seq_length:seq_length+gen_length
        total_results["chat"] = total_results["prefill"]
        for k, v in result["total_results"]["decode"].items():
            total_results["chat"][k] += v * gen_length
        for name, input_names in layer_graph.items():
            if name in result["decode"]:
                stage_results[name]["OPs"] += result["decode"][name]["OPs"] * gen_length
                stage_results[name]["memory_access"] += result["decode"][name]["memory_access"] * gen_length

    for name, input_names in layer_graph.items():
        if name in ["input", "output"]:
            OPs = 0
            memory_access = 0
            info = {}
        else:
            OPs = stage_results[name]["OPs"]
            memory_access = stage_results[name]["memory_access"]
            info = stage_results[name]
        write_to_node(name, OPs, memory_access, info, input_names)

    return nodes, edges, total_results, hardware_info
