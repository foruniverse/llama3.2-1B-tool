import torch
import sys
from transformers import pipeline
from pathlib import Path

model_id = "meta-llama/Llama-3.2-1B"
project_root = Path(__file__).parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

print(sys.path)

from scripts.utils.utils import setup_logging, ensure_dir, load_yaml_config, get_device_info


config = load_yaml_config("./configs/sft_config.yaml")
model_id = config['model']['cache_dir'] + "/AI-ModelScope/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    dtype=torch.bfloat16, 
    device_map="auto"
)

# 1. 准备符合 Llama 3.2 格式的对话
messages = [
    {"role": "user", "content": "你好，请用一句话介绍你自己。"}
]

prompt1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of json [{"name":func_name,"params_name1": params_value1, "params_name2": params_value2}]

Here is a list of functions in JSON format that you can invoke.
[ { "name": "get_stock_price", "description": "获取指定股票代码的当前实时价格", "parameters": { "type": "object", "properties": { "ticker": { "type": "string", "description": "股票代码，例如 AMD, AAPL" } }, "required": ["ticker"] } } ]
<|eot_id|><|start_header_id|>user<|end_header_id|>

帮我看看英伟达现在的股价是多少？<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.
[{"name": "GeometryPresentation.createPresentation", "description": "Initializes the GIS geometry presentation within the provided UI composite, using the given result set controller.", "parameters": {"type": "dict", "properties": {"controller": {"type": "any", "description": "The IResultSetController instance responsible for controlling the result set."}, "parent": {"type": "any", "description": "The Composite UI element where the GIS presentation will be displayed."}}, "required": ["controller", "parent"]}}]<|eot_id|><|start_header_id|>user<|end_header_id|>

Help me initialize the GIS geometry presentation in a user interface, providing a specific result set controller `mapController` and a composite UI element `mapArea` to display the GIS data?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

system_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.
[{"name": "GeometryPresentation.createPresentation", "description": "Initializes the GIS geometry presentation within the provided UI composite, using the given result set controller.", "parameters": {"type": "dict", "properties": {"controller": {"type": "any", "description": "The IResultSetController instance responsible for controlling the result set."}, "parent": {"type": "any", "description": "The Composite UI element where the GIS presentation will be displayed."}}, "required": ["controller", "parent"]}}]<|eot_id|><|start_header_id|>user<|end_header_id|>

Help me initialize the GIS geometry presentation in a user interface, providing a specific result set controller `mapController` and a composite UI element `mapArea` to display the GIS data?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


prompt_gpt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.\n\nYou are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. If available tools are not relevant in assisting with user query, just respond in natural conversational language. Don't make assumptions about what values to plug into functions. After calling & executing the functions, you will be provided with function results within <tool_response> </tool_response> XML tags. Here are the available tools:\n<tools>\n[\n{\"name\": \"independent_samples_t_test\", \"description\": \"Conducts a two-sample independent t-test and returns the t-statistic, p-value, and conclusion.\", \"parameters\": {\"type\": \"dict\", \"properties\": {\"sample1\": {\"description\": \"The first sample of observations.\", \"type\": \"List[float]\", \"default\": 0.05}, \"sample2\": {\"description\": \"The second sample of observations.\", \"type\": \"List[float]\", \"default\": 0.05}, \"alpha\": {\"description\": \"The significance level of the test. Defaults to 0.05.\", \"type\": \"float, optional\"}}}, \"required\": [\"sample1\", \"sample2\"]}\n]\n</tools>\nFor each function call return a JSON object, with the following pydantic model json schema for each:\n{\"title\": \"FunctionCall\", \"type\": \"object\", \"properties\": {\"name\": {\"title\": \"Name\", \"type\": \"string\"}, \"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}}, \"required\": [\"name\", \"arguments\"]}\nEach function call should be enclosed within <tool_call> </tool_call> XML tags.\nExample:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-dict>}\n</tool_call>\n\n"
<|eot_id|><|start_header_id|>user<|end_header_id|>
I need to compare the weights of two groups of athletes. Group X weighs [70, 72, 74, 76, 78] kg and Group Y weighs [71, 73, 75, 77, 79] kg. Perform the t-test with a significance level of 0.10.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# 2. 传入参数并执行（这里修正了日志中提到的几个参数冲突）
outputs = pipe(
    # messages, 
    # system_prompt,
    prompt_gpt,
    # max_new_tokens=256,
    max_length=2048,              # 显式设为 None，避免与 max_new_tokens 冲突
    clean_up_tokenization_spaces=False  # 消除 BPE tokenizer 的警告
)

# pipeline 返回的是一个列表，Llama 3.2 聊天模式下，最后的回答在以下路径中：
# print(outputs[0]["generated_text"][-1]["content"])
print(outputs[0]["generated_text"][len(prompt_gpt):])