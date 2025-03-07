# -*- coding: utf-8 -*-
""""""

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s',
)

logger = logging.getLogger(__name__)


def case1():
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

    agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

def try_duckduckgo():
    search_tool = DuckDuckGoSearchTool()
    print(search_tool("Who's the current president of Russia?"))

def two_agents():
    from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool
    model = HfApiModel()
    web_agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        name="web_search_agent",
        description="Runs web searches for you. Give it your query as an argument."
    )

    manager_agent = CodeAgent(
        tools=[], model=model, managed_agents=[web_agent]
    )

    manager_agent.run("Who is the CEO of Hugging Face?")

def langfuse_inspect():
    from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool
    model = HfApiModel()
    web_agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        name="web_search_agent",
        description="Runs web searches for you. Give it your query as an argument."
    )

    manager_agent = CodeAgent(
        tools=[], model=model, managed_agents=[web_agent]
    )

    manager_agent.run("Who is the CEO of Hugging Face?")

    # from smolagents import (
    #     CodeAgent,
    #     ToolCallingAgent,
    #     DuckDuckGoSearchTool,
    #     VisitWebpageTool,
    #     HfApiModel,
    # )
    #
    # model = HfApiModel(
    #     model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # )
    #
    # search_agent = ToolCallingAgent(
    #     tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    #     model=model,
    #     name="search_agent",
    #     description="This is an agent that can do web search.",
    # )
    #
    # manager_agent = CodeAgent(
    #     tools=[],
    #     model=model,
    #     managed_agents=[search_agent],
    # )
    # manager_agent.run(
    #     "How can Langfuse be used to monitor and improve the reasoning and decision-making of smolagents when they execute multi-step tasks, like dynamically adjusting a recipe based on user feedback or available ingredients?"
    # )


def try_gaia():
    import datasets
    SET = "validation"
    # eva_ds = datasets.load_dataset("gaia-benchmark/GAIA")
    eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[SET]
    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})


if __name__ == "__main__":
    # try_duckduckgo()
    # two_agents()
    # langfuse_inspect()

    try_gaia()
