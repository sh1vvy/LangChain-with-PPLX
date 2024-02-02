from openai import OpenAI
from config import MY_API_KEY

messages = [
  {
    "role":"system",
    "content": (
      "Be precise and concise."
    ),
  },
  {
    "role":"user",
    "content": (
      "What are the news headlines today for India?"
    ),
  },
]

client = OpenAI(api_key=MY_API_KEY,base_url="https://api.perplexity.ai")

# chat completion without streaming
response = client.chat.completions.create(
    model="pplx-7b-online",
    messages=messages,
)
print(response)