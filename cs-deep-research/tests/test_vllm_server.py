# -*- coding: utf-8 -*-
""""""


# vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype auto --api-key token-abc123

import logging

logger = logging.getLogger(__name__)

from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="Qwen/Qwen2.5-0.5B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)