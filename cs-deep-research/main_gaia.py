import json
import logging
import os
import random
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import datasets
import hydra
import numpy as np
import pandas as pd
import torch
from cs_deep_research.helpers.reformulator import prepare_response
from cs_deep_research.helpers.run_agents import (
    get_single_file_description,
    get_zip_description,
)
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
)
from cs_deep_research.helpers.visual_qa import visualizer
from cs_deep_research.templates import PROMPTS
from cs_deep_research.utils import get_id_func
from huggingface_hub import login
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from smolagents import (
    CodeAgent,
    Model,
    # HfApiModel,
    OpenAIServerModel,
    ToolCallingAgent,
)


logger = logging.getLogger(__name__)

login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()
OmegaConf.register_new_resolver("generate_job_id", get_id_func())

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


@hydra.main(version_base=None, config_path="configs", config_name="config_gaia")
def main(cfg: DictConfig) -> None:
    if cfg.seed:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Exp Dir: {HydraConfig.get().run.dir}")

    eval_dataset = get_eval_dataset(
        hf_path=cfg.dataset.hf_path,
        split=cfg.dataset.split,
        data_dir=cfg.dataset.data_dir,
    )
    # TODO, if one task is done before, we can skip it, refer `get_examples_to_answer` in `run_gaia.py`
    task_to_run = eval_dataset.to_list()

    records = []
    if cfg.concurrency.mode == "linear":
        for i, one_task in enumerate(task_to_run):
            difficulty = one_task["difficulty"]
            if difficulty not in str(cfg.dataset.level):
                logger.warning(f"Skip task with difficulty {difficulty}")
                continue
            logger.info(f"Start running task {one_task} with difficulty {difficulty}")
            task_record = answer_one_task(
                cfg=cfg,
                example=one_task,
                llm_model_id=cfg.llm_model.model_id,
                provider=cfg.llm_model.provider,
                api_base=cfg.llm_model.localhost.api_base,
                api_key=cfg.llm_model.localhost.api_key,
                visual_inspection_tool=visualizer,
                text_inspector_max_limit=cfg.text_inspector_max_limit,
            )
            records.append(task_record)

            if cfg.debug.enable and i >= cfg.debug.max_tasks:
                logger.critical(f"Debug mode enabled, stop after {cfg.debug.max_tasks} tasks")
                break

        # Save the records
        records_path = Path(HydraConfig.get().run.dir) / "records.json"
        with open(records_path, "w") as f:
            for record in records:
                f.write(json.dumps(record.__dict__) + "\n")
        logger.info(f"Records saved to {records_path}")
    else:
        raise NotImplementedError


@dataclass
class TaskRecord:
    question: str
    augmented_question: str
    difficulty: str
    task_id: str
    prediction: str
    true_answer: str
    agent_llm_name: str
    intermediate_steps: List[str]
    parsing_error: bool
    iteration_limit_exceeded: bool
    agent_error: str
    start_time: str
    end_time: str


def answer_one_task(
    cfg: DictConfig,
    example: dict,
    llm_model_id: str,
    provider: str,
    api_base: str,
    api_key: str,
    visual_inspection_tool: Callable,
    text_inspector_max_limit: int = 100000,
) -> TaskRecord:
    if provider == "localhost":
        llm_model = OpenAIServerModel(
            model_id=llm_model_id,
            api_base=api_base,
            api_key=api_key,
        )
    else:
        raise NotImplementedError(f"Provider {provider} not implemented.")

    document_inspection_tool = TextInspectorTool(llm_model, text_inspector_max_limit)

    agent_system = create_agent_hierarchy(
        llm_model,
        browser_config=cfg.browser_config,
        search_agent_config=cfg.search_agent,
        manager_agent_config=cfg.manager_agent,
        text_inspector_max_limit=text_inspector_max_limit,
    )

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
    Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
    Run verification steps if that's needed, you must make sure you find the correct answer!
    Here is the task:
    """ + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
            prompt_use_files += get_single_file_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent_system.run(augmented_question)
        agent_memory = agent_system.write_memory_to_messages(summary_mode=True)
        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=llm_model)
        output = str(final_result)
        for memory_step in agent_system.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = [str(step) for step in agent_system.memory.steps]
        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False
        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_record = TaskRecord(
        question=example["question"],
        augmented_question=augmented_question,
        difficulty=example["difficulty"],
        task_id=example["task_id"],
        prediction=output,
        true_answer=example["true_answer"],
        agent_llm_name=llm_model_id,
        intermediate_steps=intermediate_steps,
        parsing_error=parsing_error,
        iteration_limit_exceeded=iteration_limit_exceeded,
        agent_error=str(exception) if raised_exception else None,
        start_time=start_time,
        end_time=end_time,
    )
    return task_record


def create_agent_hierarchy(
    llm_model: Model,
    browser_config: DictConfig,
    search_agent_config: DictConfig,
    manager_agent_config: DictConfig,
    text_inspector_max_limit: int = 100000,
) -> CodeAgent:
    text_inspector_tool = TextInspectorTool(llm_model, text_inspector_max_limit)
    browser_kwargs = OmegaConf.to_container(browser_config, resolve=True)
    browser_kwargs["request_kwargs"]["headers"] = {"User-Agent": browser_config.request_kwargs.headers.user_agent}
    browser_kwargs["serpapi_key"] = os.getenv(browser_kwargs["serpapi_key"])
    browser = SimpleTextBrowser(**browser_kwargs)

    web_tools = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(llm_model, text_inspector_max_limit),
    ]

    search_agent = ToolCallingAgent(
        model=llm_model,
        tools=web_tools,
        max_steps=search_agent_config.max_steps,
        verbosity_level=search_agent_config.verbosity_level,
        planning_interval=search_agent_config.planning_interval,
        name=search_agent_config.name,
        description=PROMPTS[search_agent_config.description],
        provide_run_summary=search_agent_config.provide_run_summary,
    )
    search_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=llm_model,
        tools=[visualizer, text_inspector_tool],
        max_steps=manager_agent_config.max_steps,
        verbosity_level=manager_agent_config.verbosity_level,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=manager_agent_config.planning_interval,
        managed_agents=[search_agent],
        name=manager_agent_config.name,
    )
    return manager_agent


def get_eval_dataset(hf_path: str, split: str, data_dir: str) -> datasets.Dataset:
    """
    Load the evaluation dataset
    :param hf_path: gaia-benchmark/GAIA
    :param split: validation or test
    :param data_dir: the directory where the data is stored
    :return: the evaluation dataset
    """
    eval_dataset = datasets.load_dataset(path=hf_path, name="2023_all")[split]
    eval_dataset = eval_dataset.rename_columns(
        {"Question": "question", "Final answer": "true_answer", "Level": "difficulty"}
    )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"{data_dir}/{split}/" + row["file_name"]
        return row

    eval_dataset = eval_dataset.map(preprocess_file_paths)
    logger.info(f"Loaded {len(eval_dataset)} examples from {split} set")
    eval_df = pd.DataFrame(eval_dataset)
    logger.info(f"Difficulty (level) Counts: {eval_df['difficulty'].value_counts()}")
    return eval_dataset


if __name__ == "__main__":
    main()
