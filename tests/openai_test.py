from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
#   // extra_headers={
#   //   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#   //   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   // },
  extra_body={},
  model="x-ai/grok-code-fast-1",
  messages=[
              {
                "role": "user",
                "content": "What is the meaning of life?"
              }
            ]
)
print(completion.choices[0].message.content)