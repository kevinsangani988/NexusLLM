from litellm import completion

response = completion(
  model="ollama_chat/qwen2.5:1.5b",
  messages=[{"role": "user", "content": "Hello, how are you?"}],
  api_base="http://localhost:11434"
)
print(response.choices[0].message.content)