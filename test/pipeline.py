import sys
import json
import torch
from pathlib import Path
from threading import Thread
from transformers import pipeline, TextIteratorStreamer

model_id = "meta-llama/Llama-3.2-1B"
project_root = Path(__file__).parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

print(sys.path)

from scripts.utils.utils import setup_logging, ensure_dir, load_yaml_config, get_device_info


def build_bfcl_prompt(user_question: str, tools_list: list, system_prompt: str = None) -> str:
    """
    组装符合 Llama 3.2 规范的 BFCL 评测 Prompt
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert in composing functions. You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
            "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n"
            "You should only return the function calls in your response.\n\n"
            "If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n"
            "You SHOULD NOT include any other text in the response.\n\n"
            "Here is a list of functions in JSON format that you can invoke."
        )
    
    tools_json_str = json.dumps(tools_list, ensure_ascii=False)
    full_system = f"{system_prompt}\n{tools_json_str}"
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{full_system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt

# 1. 定义不同的 Tools 与问题并组装
my_tools = [{
    "name": "GeometryPresentation.createPresentation",
    "description": "Initializes the GIS geometry presentation within the provided UI composite...",
    "parameters": {
        "type": "dict",
        "properties": {
            "controller": {"type": "any", "description": "The controller instance."},
            "parent": {"type": "any", "description": "The Composite UI element."}
        },
        "required": ["controller", "parent"]
    }
}]

my_question = "Help me initialize the GIS geometry presentation in a user interface, providing a specific result set controller `mapController` and a composite UI element `mapArea` to display the GIS data?"

final_prompt = build_bfcl_prompt(user_question=my_question, tools_list=my_tools)

# 2. 初始化模型与配置
config = load_yaml_config("./configs/sft_config.yaml")
model_id = str(project_root / "models/checkpoints/sft/checkpoint-471")

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    dtype=torch.bfloat16, 
    device_map="auto"
)

# 3. 创建 Streamer 并设置 skip_prompt=True 自动过滤输入的 Prompt
streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, clean_up_tokenization_spaces=False)

# 4. 可调节的生成参数字典
generation_kwargs = dict(
    text_inputs=final_prompt,
    streamer=streamer,
    max_new_tokens=2048,
    
    # 常用调节参数：
    do_sample=True,             # 开启采样以使用 temperature 和 top_p
    temperature=0.1,            # 降低随机性使其严格按格式输出（评测建议设置较小，如 0.1）
    top_p=0.9,                  # 核采样阈值
    repetition_penalty=1.05,    # 重复惩罚系数
    
    clean_up_tokenization_spaces=False
)

# 5. 在子线程中启动生成任务（避免主线程因等待文本生成而阻塞流式输出）
thread = Thread(target=pipe, kwargs=generation_kwargs)
thread.start()

# 6. 主线程实时迭代读取并打印流式内容
print("模型输出: ", end="", flush=True)
for new_text in streamer:
    print(new_text, end="", flush=True)
print()  # 换行

# 确保线程完全结束
thread.join()