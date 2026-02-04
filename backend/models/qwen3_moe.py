
def get_num_attention_heads(model_params):
    return model_params["num_attention_heads"]

def get_hidden_size(model_params):
    return model_params["hidden_size"]

def get_num_key_value_heads(model_params):
    return model_params["num_key_value_heads"]

def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]

def get_num_hidden_layers(model_params):
    return model_params["num_hidden_layers"]

def get_intermediate_size(model_params):
    return model_params["intermediate_size"]

def get_vocab_size(model_params):
    return model_params["vocab_size"]
    
def get_num_active_experts(model_params):
    return model_params["num_experts_per_tok"]

def post_process(model_params,args):
    hiddensize=get_hidden_size(model_params)
    vocab_size=get_vocab_size(model_params)
    layers=[]

    layers.append({
        'name': 'lm_head',
        'stage': "prefill",
        'OPs': args['batchsize'] * args['seqlen'] * hiddensize * vocab_size * 2,
        'load_weight': hiddensize * vocab_size * args['w_byte'],
        'load_act': args['batchsize'] * args['seqlen'] * hiddensize * args['a_byte'],
        'store_act': args['batchsize'] * args['seqlen'] * vocab_size * args['a_byte'],
    })

    layers.append({
        'name': 'lm_head',
        'stage': "decode",
        'OPs': args['batchsize'] * hiddensize * vocab_size * 2,
        'load_weight': hiddensize * vocab_size * args['w_byte'],
        'load_act': args['batchsize'] * hiddensize * args['a_byte'],
        'store_act': args['batchsize'] * vocab_size * args['a_byte'],
    })
    return layers

def get_linear_layers(model_params, tp_size: int):
    hidden_size=get_hidden_size(model_params)
    intermediate_size=get_intermediate_size(model_params)
    key_value_heads=get_num_key_value_heads(model_params)
    attention_heads=get_num_attention_heads(model_params)
    
    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
        assert key_value_heads % tp_size == 0
    
    return {
        "q_proj":[hidden_size, hidden_size // tp_size],
        "k_proj":[hidden_size, hidden_size * key_value_heads // attention_heads // tp_size],
        "v_proj":[hidden_size, hidden_size * key_value_heads // attention_heads // tp_size],
        "out_proj":[hidden_size // tp_size, hidden_size],
        "gate_proj":[hidden_size, intermediate_size // tp_size],
        "up_proj":[hidden_size,intermediate_size // tp_size],
        "down_proj":[intermediate_size // tp_size, hidden_size],
    }

#TODO: moe models have different graph structure
from models.llama import flashattention_transformer_layer_graph,transformer_layer_graph