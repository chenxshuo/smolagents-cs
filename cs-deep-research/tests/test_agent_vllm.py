# -*- coding: utf-8 -*-
""""""

import logging

from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel, VLLMModel


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)


def case1():
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=VLLMModel(model_id="Qwen/Qwen2.5-0.5B-Instruct"))
    agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")


def vllm_server():
    model = OpenAIServerModel(
        model_id="Qwen/QwQ-32B",
        api_base="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
    agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")


if __name__ == "__main__":
    # case1()
    vllm_server()
