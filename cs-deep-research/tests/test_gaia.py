from smolagents import api_key_setup

import datasets
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")


# eva_ds = datasets.load_dataset("gaia-benchmark/GAIA")
eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")["validation"]
eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
print(eval_ds[0])