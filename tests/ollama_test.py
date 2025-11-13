import ollama

stream = ollama.chat(model='qwen2.5-coder:7b', stream=True, messages=[
    {'role': 'user', 'content': 'Напиши функцию сортировки пузырьком на Python.'}
])

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
