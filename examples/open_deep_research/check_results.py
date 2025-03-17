# -*- coding: utf-8 -*-
""""""

import json
import logging


logger = logging.getLogger(__name__)


def check_results(log_path):
    total = 0
    correct = 0
    with open(log_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["prediction"] and data["true_answer"]:
                total += 1
                if data["prediction"].lower() == data["true_answer"].lower():
                    correct += 1
                    print(f"CORRECT: prediction: {data['prediction']}, true_answer: {data['true_answer']}")
                else:
                    print(f"WRONG: prediction: {data['prediction']}, true_answer: {data['true_answer']}")
            else:
                logger.error(f"Missing prediction or true_answer in {data}")

    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total}")


if __name__ == "__main__":
    # log_path = "./output/validation/gaia-level-1.jsonl"
    # log_path = "./output/validation/gaia-level1-qwq-32b.jsonl"
    log_path = "./output/validation/gaia-level-1-ds-llama70b.jsonl"
    # log_path = "./output/validation/gaia-level-1-ds-qwen32b.jsonl"
    check_results(log_path)
