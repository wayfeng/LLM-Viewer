import math


def _text_config(model_params):
	return model_params.get("text_config", model_params)


def _vision_config(model_params):
	return model_params.get("vision_config", {})


# ===== Text branch (LLM) =====
def get_num_attention_heads(model_params):
	return _text_config(model_params)["num_attention_heads"]


def get_hidden_size(model_params):
	return _text_config(model_params)["hidden_size"]


def get_head_dim(model_params):
	text_config = _text_config(model_params)
	head_dim = text_config.get("head_dim")
	if head_dim is None:
		head_dim = text_config["hidden_size"] // text_config["num_attention_heads"]
	return head_dim


def get_num_key_value_heads(model_params):
	return _text_config(model_params)["num_key_value_heads"]


def get_norm_layers(model_params):
	return ["attn_norm", "mlp_norm"]


def get_num_hidden_layers(model_params):
	return _text_config(model_params)["num_hidden_layers"]


def get_intermediate_size(model_params):
	return _text_config(model_params)["intermediate_size"]


def get_vocab_size(model_params):
	return _text_config(model_params)["vocab_size"]


def get_linear_layers(model_params):
	text_config = _text_config(model_params)
	hidden_size = text_config["hidden_size"]
	head_dim = get_head_dim(model_params)
	intermediate_size = text_config["intermediate_size"]
	key_value_heads = text_config["num_key_value_heads"]
	attention_heads = text_config["num_attention_heads"]

	return {
		"q_proj": [hidden_size, attention_heads * head_dim],
		"k_proj": [hidden_size, key_value_heads * head_dim],
		"v_proj": [hidden_size, key_value_heads * head_dim],
		"out_proj": [attention_heads * head_dim, hidden_size],
		"gate_proj": [hidden_size, intermediate_size],
		"up_proj": [hidden_size, intermediate_size],
		"down_proj": [intermediate_size, hidden_size],
	}


# ===== Vision branch =====
def get_vision_num_heads(model_params):
	return _vision_config(model_params)["num_heads"]


def get_vision_hidden_size(model_params):
	return _vision_config(model_params)["hidden_size"]


def get_vision_intermediate_size(model_params):
	return _vision_config(model_params)["intermediate_size"]


def get_vision_num_hidden_layers(model_params):
	return _vision_config(model_params)["depth"]


def get_vision_patch_size(model_params):
	return _vision_config(model_params)["patch_size"]


def get_vision_in_channels(model_params):
	return _vision_config(model_params)["in_channels"]


def get_vision_out_hidden_size(model_params):
	return _vision_config(model_params).get("out_hidden_size")


def get_vision_spatial_merge_size(model_params):
	return _vision_config(model_params).get("spatial_merge_size", 1)


def get_vision_temporal_patch_size(model_params):
	return _vision_config(model_params).get("temporal_patch_size", 1)


def get_vision_norm_layers(model_params):
	return ["vision_norm1", "vision_norm2"]


def get_vision_linear_layers(model_params):
	hidden_size = get_vision_hidden_size(model_params)
	intermediate_size = get_vision_intermediate_size(model_params)
	attention_heads = get_vision_num_heads(model_params)
	head_dim = hidden_size // attention_heads

	return {
		"vision_q_proj": [hidden_size, attention_heads * head_dim],
		"vision_k_proj": [hidden_size, attention_heads * head_dim],
		"vision_v_proj": [hidden_size, attention_heads * head_dim],
		"vision_out_proj": [attention_heads * head_dim, hidden_size],
		"vision_gate_proj": [hidden_size, intermediate_size],
		"vision_up_proj": [hidden_size, intermediate_size],
		"vision_down_proj": [intermediate_size, hidden_size],
	}

def get_vision_repeat_layers():
	return {
		"vision_q_proj",
		"vision_k_proj",
		"vision_v_proj",
		"vision_out_proj",
		"vision_gate_proj",
		"vision_up_proj",
		"vision_down_proj",
		"vision_qk_matmul",
		"vision_sv_matmul",
		"vision_softmax",
		"vision_norm1",
		"vision_norm2",
		"vision_attn_add",
		"vision_mlp_add",
		"vision_mlp_act",
		"vision_fused_attention",
	}

def vision_post_process(model_params, args):
	out_hidden_size = get_vision_out_hidden_size(model_params)
	hidden_size = get_vision_hidden_size(model_params)
	if out_hidden_size is None:
		return []
	return [{
		"name": "vision_proj",
		"stage": "vision",
		"OPs": args["batchsize"] * args["seqlen"] * hidden_size * out_hidden_size * 2,
		"load_weight": hidden_size * out_hidden_size * args["w_byte"],
		"load_act": args["batchsize"] * args["seqlen"] * hidden_size * args["a_byte"],
		"store_act": args["batchsize"] * args["seqlen"] * out_hidden_size * args["a_byte"],
	}]


# ===== Layer graphs =====
transformer_layer_graph = {
	"input": [],
	"attn_norm": ["input"],
	"q_proj": ["attn_norm"],
	"k_proj": ["attn_norm"],
	"v_proj": ["attn_norm"],
	"qk_matmul": ["q_proj", "k_proj"],
	"softmax": ["qk_matmul"],
	"sv_matmul": ["softmax", "v_proj"],
	"out_proj": ["sv_matmul"],
	"attn_add": ["input", "out_proj"],
	"mlp_norm": ["attn_add"],
	"gate_proj": ["mlp_norm"],
	"up_proj": ["mlp_norm"],
	"mlp_act": ["up_proj", "gate_proj"],
	"down_proj": ["mlp_act"],
	"mlp_add": ["attn_add", "down_proj"],
	"output": ["mlp_add"],
}

flashattention_transformer_layer_graph = {
	"input": [],
	"attn_norm": ["input"],
	"q_proj": ["attn_norm"],
	"k_proj": ["attn_norm"],
	"v_proj": ["attn_norm"],
	"fused_attention": ["q_proj", "k_proj", "v_proj"],
	"out_proj": ["fused_attention"],
	"attn_add": ["input", "out_proj"],
	"mlp_norm": ["attn_add"],
	"gate_proj": ["mlp_norm"],
	"up_proj": ["mlp_norm"],
	"mlp_act": ["up_proj", "gate_proj"],
	"down_proj": ["mlp_act"],
	"mlp_add": ["attn_add", "down_proj"],
	"output": ["mlp_add"],
}

vision_layer_graph = {
	"vision_input": [],
	"vision_patch_embed": ["vision_input"],
	"vision_norm1": ["vision_patch_embed"],
	"vision_q_proj": ["vision_norm1"],
	"vision_k_proj": ["vision_norm1"],
	"vision_v_proj": ["vision_norm1"],
	"vision_qk_matmul": ["vision_q_proj", "vision_k_proj"],
	"vision_softmax": ["vision_qk_matmul"],
	"vision_sv_matmul": ["vision_softmax", "vision_v_proj"],
	"vision_out_proj": ["vision_sv_matmul"],
	"vision_attn_add": ["vision_patch_embed", "vision_out_proj"],
	"vision_norm2": ["vision_attn_add"],
	"vision_gate_proj": ["vision_norm2"],
	"vision_up_proj": ["vision_norm2"],
	"vision_mlp_act": ["vision_up_proj", "vision_gate_proj"],
	"vision_down_proj": ["vision_mlp_act"],
	"vision_mlp_add": ["vision_attn_add", "vision_down_proj"],
	"vision_proj": ["vision_mlp_add"],
	"vision_output": ["vision_proj"],
}

vision_flashattention_layer_graph = {
	"vision_input": [],
	"vision_patch_embed": ["vision_input"],
	"vision_norm1": ["vision_patch_embed"],
	"vision_q_proj": ["vision_norm1"],
	"vision_k_proj": ["vision_norm1"],
	"vision_v_proj": ["vision_norm1"],
	"vision_fused_attention": ["vision_q_proj", "vision_k_proj", "vision_v_proj"],
	"vision_out_proj": ["vision_fused_attention"],
	"vision_attn_add": ["vision_patch_embed", "vision_out_proj"],
	"vision_norm2": ["vision_attn_add"],
	"vision_gate_proj": ["vision_norm2"],
	"vision_up_proj": ["vision_norm2"],
	"vision_mlp_act": ["vision_up_proj", "vision_gate_proj"],
	"vision_down_proj": ["vision_mlp_act"],
	"vision_mlp_add": ["vision_attn_add", "vision_down_proj"],
	"vision_proj": ["vision_mlp_add"],
	"vision_output": ["vision_proj"],
}
