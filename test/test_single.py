from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/yanyan/project/llama3.2-1B-tool/models/sft-360"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, clean_up_tokenization_spaces=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

system_content = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of:
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]

You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.

[
  {
    "name": "genetics.calculate_similarity",
    "description": "Calculates the genetic similarity between two species.",
    "parameters": {
      "type": "dict",
      "properties": {
        "format": {
          "type": "string",
          "description": "The format of the result (percentage or fraction). Default is percentage."
        },
        "species2": {
          "type": "string",
          "description": "The second species to compare."
        },
        "species1": {
          "type": "string",
          "description": "The first species to compare."
        },

      },
      "required": [
        "species1",
        "species2"
      ]
    }
  }
]
"""

user_content = "Find out how genetically similar a human and a chimp are in percentage."
user_content = "Compare the genetic similarity between species human and species chimp."
user_content = "Compare the genetic similarity between species1 human and species2 chimp"
user_content = "Compare the genetic similarity between species1 human and species2 chimp in percentage"


system_content = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.

[
  {
    "name": "todo",
    "description": "Manages a todo list allowing the user to add, delete, or update items. Note that the provided function is in Python 3 syntax.",
    "parameters": {
      "type": "dict",
      "required": [
        "x",
        "y"
      ],
      "properties": {
        "x"on": {
          "type": "string",
          "description": "The action to be performed on the todo list.",
          "enum": [
            "add",
            "delete",
            "update"
          ]
        },
        "y": {
          "type": "string",
          "description": "The details of the todo item relevant to the action being performed."
        }
      }
    }
  }
]
<|eot_id|>
"""

user_content = "Hi there! Could you please help me manage my tasks? I need to add a task called 'Machine Learning Study Session'. Also, I have completed one of my tasks named 'todo random', and I would like to delete it from my list."

messages = [
    {
        "role": "system",
        "content": system_content,
    },
    {
        "role": "user",
        "content": user_content,
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(prompt)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.0,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
        pad_token_id=tokenizer.eos_token_id,
    )

gen = output[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(gen, skip_special_tokens=False))