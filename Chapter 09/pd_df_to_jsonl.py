import pandas as pd
from sklearn.datasets import fetch_20newsgroups

categories = ["rec.sport.baseball", "rec.sport.hockey"]
sports_dataset = fetch_20newsgroups(
    subset="train", shuffle=True, random_state=42, categories=categories
)
len_all, len_baseball, len_hockey = (
    len(sports_dataset.data),
    len([e for e in sports_dataset.target if e == 0]),
    len([e for e in sports_dataset.target if e == 1]),
)
print(
    f"Total examples: {len_all}, Baseball examples: {len_baseball}, Hockey examples: {len_hockey}"
)
labels = [
    sports_dataset.target_names[x].split(".")[-1] for x in sports_dataset["target"]
]
texts = [text.strip() for text in sports_dataset["data"]]
df = pd.DataFrame(zip(texts, labels), columns=["prompt", "completion"])  # [:300]
df.head()
df.to_json("sport2.jsonl", orient="records", lines=True)
