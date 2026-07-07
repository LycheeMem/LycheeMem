# test openai chat api
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")

response = client.chat.completions.create(
    model="lycheemem",
    messages=[
        {"role": "user", "content": "Melanie是谁"}
    ]
)