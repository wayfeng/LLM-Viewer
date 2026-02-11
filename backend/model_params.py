import requests
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


available_model_ids_sources = {
    "Qwen/Qwen3-4B-Instruct-2507": {"source": "huggingface"},
    "Qwen/Qwen3-32B": {"source": "huggingface"},
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {"source": "huggingface"},
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {"source": "huggingface"},
    "openai/gpt-oss-20b": {"source": "huggingface"},
    "openai/gpt-oss-120b": {"source": "huggingface"},
    "LLM-Research/llama-2-7b": {"source": "modelscope"},
    "LLM-Research/llama-2-13b": {"source": "modelscope"},
    "zai-org/chatglm3-6b": {"source": "huggingface"},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {"source": "huggingface"},
}

def get_available_models():
    return list(available_model_ids_sources.keys())

def get_model_config_path(model_id: str):
    model_name = model_id.split("/")[-1].lower().replace('.', '_')
    return f"./models/{model_name}/config.json"

def download_model_config(model_id: str, source: str = "huggingface"):
    #model_id = "Qwen/Qwen2.5-7B"
    path = Path(get_model_config_path(model_id)).parent
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    file_name = "config.json"
    if source == "huggingface":
        url = f"https://huggingface.co/{model_id}/resolve/main/{file_name}"
    elif source == "modelscope":
        url = f"https://www.modelscope.cn/api/v1/models/{model_id}/repo?Revision=master&FilePath={file_name}"
    else:
        raise ValueError(f"Unsupported source: {source}")

    response = requests.get(url, allow_redirects=True)

    if response.status_code == 200:
        with open(path / file_name, "wb") as f:
            f.write(response.content)
        logger.info(f"Successfully downloaded {file_name} from {source}")
        return str(path / file_name)
    else:
        logger.error(f"Failed to download. Status code: {response.status_code}")
    return None

def load_model_params(model_id: str, config_file: str = None):
    if config_file is None:
        config_file = get_model_config_path(model_id)
    if not Path(config_file).exists():
        source = available_model_ids_sources[model_id]["source"]
        config_file = download_model_config(model_id, source)
    with open(config_file, "r") as f:
        model_params = json.load(f)
    return model_params


class AbstractModelConfig:
    def get_num_attention_heads(self, model_params):
        raise NotImplementedError

    def get_hidden_size(self, model_params):
        raise NotImplementedError

    def get_head_dim(self, model_params):
        raise NotImplementedError

    def get_num_key_value_heads(self, model_params):
        raise NotImplementedError

    def get_norm_layers(self, model_params):
        raise NotImplementedError

    def get_num_hidden_layers(self, model_params):
        raise NotImplementedError

    def get_intermediate_size(self, model_params):
        raise NotImplementedError

    def get_vocab_size(self, model_params):
        raise NotImplementedError

    def post_process(self, model_params, args):
        raise NotImplementedError

    def get_linear_layers(self, model_params, tp_size: int):
        raise NotImplementedError