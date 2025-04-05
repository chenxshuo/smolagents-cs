
import logging
import os
from smolagents import OpenAIServerModel, ToolCallingAgent
from cs_deep_research.helpers.text_inspector_tool import TextInspectorTool
from cs_deep_research.helpers.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
    PostTool
)
from cs_deep_research import api_key_setup
from smolagents.agents import CodeAgent 

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]

def prepare_browser_agent(model):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

    BROWSER_CONFIG = {
        "viewport_size": 1024 * 5,
        "downloads_folder": "downloads_folder",
        "request_kwargs": {
            "headers": {"User-Agent": user_agent},
            "timeout": 300,
        },
        "serpapi_key": os.getenv("SERPAPI_API_KEY"),
    }

    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    web_tools = [
        # SearchInformationTool(browser),
        VisitTool(browser),
        # PageUpTool(browser),
        # PageDownTool(browser),
        # FinderTool(browser),
        # FindNextTool(browser),
        # ArchiveSearchTool(browser),
    ]
    text_browser_agent =  ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_browser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    return browser, text_browser_agent



def test_access_specific_website():
    website_link = "https://kidlanli.github.io/Agent-Slack-off/"
    model = OpenAIServerModel(
        model_id="Qwen/QwQ-32B",
        api_base="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    browser, text_browser_agent = prepare_browser_agent(model)
    agent = CodeAgent(
        model=model,
        # tools=[TextInspectorTool(model, 100000)],
        tools = [VisitTool(browser), PostTool(browser)],
        max_steps=4,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        # managed_agents=[browser_agent],
    )
    result = agent.run(f"Access the website {website_link}, read through its content and follow the instruction there.")



if __name__ == "__main__":
    test_access_specific_website()

    
