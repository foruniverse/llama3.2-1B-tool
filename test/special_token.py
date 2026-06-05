import torch
import sys
from transformers import AutoTokenizer, pipeline
from pathlib import Path

model_id = "meta-llama/Llama-3.2-1B"
project_root = Path(__file__).parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

from scripts.utils.utils import setup_logging, ensure_dir, load_yaml_config, get_device_info



config = load_yaml_config("./configs/sft_config.yaml")
model_id = config['model']['cache_dir'] + "/AI-ModelScope/Llama-3.2-1B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print(tokenizer.pad_token_id, tokenizer.pad_token)
print(tokenizer.eos_token_id, tokenizer.eos_token)
print(tokenizer.all_special_tokens)
# print(tokenizer.chat_template)
# print(tokenizer.special_tokens_map)
# print(tokenizer.added_tokens_decoder)