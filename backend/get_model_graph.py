from model_analyzer import get_analyzer
from utils import str_number, print_params

@print_params
def get_model_graph(model_id, inference_config):
    use_flashattention = bool(inference_config["use_flashattention"])
    genlen = int(inference_config["genlen"])

    configs = {}
    for key in inference_config.keys():
        if key not in ["stage", "genlen"]:
            configs[key] = inference_config[key]

    analyzer = get_analyzer(model_id)
    result = analyzer.analyze(**inference_config)

    GQA = analyzer.if_group_qa()

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

    stage = inference_config["stage"]
    total_results = result["total_results"]
    stage_results = result[stage]

    if use_flashattention:
        layer_graph = analyzer.module.flashattention_transformer_layer_graph
    else:
        layer_graph = analyzer.module.transformer_layer_graph
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

    return nodes, edges, total_results
