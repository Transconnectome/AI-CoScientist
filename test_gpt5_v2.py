from openai import OpenAI

with open('.env', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            api_key = line.strip().split('=', 1)[1]
            break

client = OpenAI(api_key=api_key)

print("Testing GPT-5 with longer max_completion_tokens...")
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a 50-word paragraph about scientific writing."}
    ],
    max_completion_tokens=500
)

print(f"\nResponse ({len(response.choices[0].message.content.split())} words):")
print(response.choices[0].message.content)
print(f"\nFinish reason: {response.choices[0].finish_reason}")
