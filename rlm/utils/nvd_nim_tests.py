from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-v4-pro",
  messages=[{"role":"user","content":"Explain how to find thevenin equivalent of a circuit"}],
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  extra_body={"chat_template_kwargs":{"thinking":False}},
  stream=True
)

for chunk in completion:
  if not getattr(chunk, "choices", None):
    continue
  if chunk.choices and chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")