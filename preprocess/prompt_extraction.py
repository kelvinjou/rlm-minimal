from datasets import load_dataset
import pandas as pd


# ds = load_dataset("ai-safety-institute/AgentHarm", "chat")

# print(ds)

# df = ds["test_public"].to_pandas()

# df.to_csv("preprocess/chat_test_public.csv", index=False)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("oolongbench/oolong-real", "toy_dnd")

df = ds["validation"].select(range(1)).to_pandas()
df.to_csv("preprocess/oolong-toy-validation-head-1.csv", index=False)