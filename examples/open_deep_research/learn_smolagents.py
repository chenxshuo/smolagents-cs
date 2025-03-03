# -*- coding: utf-8 -*-
""""""

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
import api_key_setup

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

if __name__ == "__main__":
    # try_duckduckgo()
    two_agents()