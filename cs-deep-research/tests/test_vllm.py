# -*- coding: utf-8 -*-
""""""

import logging
from vllm import LLM, SamplingParams
from smolagents import api_key_setup
import os

logger = logging.getLogger(__name__)
# SPDX-License-Identifier: Apache-2.0

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# print("HF_HOME", os.environ.get("HF_HOME"))

# Create an LLM.
# llm = LLM(model="/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6")

# llm = LLM(model="facebook/opt-125m")
model = "Qwen/Qwen2.5-72B-Instruct"
# model = "Qwen/QwQ-32B"
llm = LLM(model=model, tensor_parallel_size=2)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
