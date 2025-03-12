from smolagents import api_key_setup

import datasets
from collections import Counter
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")


# eva_ds = datasets.load_dataset("gaia-benchmark/GAIA")
eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")["validation"]
eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
import ipdb; ipdb.set_trace()


levelcount = Counter(eval_ds["task"]) # 2:86, 1:53, 3:26
print(levelcount)
print(eval_ds[0])