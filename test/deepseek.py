from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="sk-cd5d2eb6225e4cd6ba50afa143be0366",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function for quicksort."
        }
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)