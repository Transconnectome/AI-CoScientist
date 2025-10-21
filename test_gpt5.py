import os
from openai import OpenAI

# Load API key
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            api_key = line.strip().split('=', 1)[1]
            break

client = OpenAI(api_key=api_key)

print("Testing GPT-5...")
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, GPT-5 is working!' in one sentence."}
    ],
    max_completion_tokens=100
)

print(f"Response: {response.choices[0].message.content}")
print(f"Model: {response.model}")
print(f"Finish reason: {response.choices[0].finish_reason}")
