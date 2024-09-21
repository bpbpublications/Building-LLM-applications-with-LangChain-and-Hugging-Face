"""
This script will show how to use different evaluation matrices to validate the models
and output.
Please note that for open source models you dont need to provide token.

https://huggingface.co/docs/evaluate/a_quick_tour
https://huggingface.co/evaluate-metric
https://huggingface.co/evaluate-measurement
https://huggingface.co/evaluate-comparison
"""

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

# Define the token
token = "PUT_HUGGINGFACE_TOKEN_HERE"

Q1 = "Explain Large Language Models in funny way so that child can understand."
Q2 = "What is cricket provide brief details."

# Load the data on which databricks/dolly-v2-3b model has been trained
dolly_dataset = load_dataset(
    "databricks/databricks-dolly-15k",
    cache_dir="E:\\Repository\\Book\\data_cache",
    token=token,
)

# load the responses from the data.
dolly_response_data = [k for k in dolly_dataset["train"]["response"]]

# Load the model from local system - Model -1 ..........................................................................
dolly_generate_text = pipeline(
    model="databricks/dolly-v2-3b",
    trust_remote_code=True,
    device_map="auto",  # make it "auto" for auto selection between GPU and CPU, -1 for CPU, 0 for GPU
    return_full_text=True,  # necessary to return complete text.
    tokenizer=AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", token=token),
    model_kwargs={
        "max_length": 100,  # generate this number of tokens
        # change the cache_dir based on your preferences
        "cache_dir": "E:\\Repository\\Book\\models",
        "offload_folder": "offload",  # use it when model size is > 7B
    },
)

# get the answer of the question - 1
dl_ans_1 = dolly_generate_text(Q1)

# get the answer of the question - 2
dl_ans_2 = dolly_generate_text(Q2)

# ======================================================================================================================
"""
ROUGE SCORE
The ROUGE values are in the range of 0 to 1.

HIGHER the score better the result

IN THE OUTPUT...........................................
"rouge1": unigram (1-gram) based scoring - The model recalled X% of the single words from the reference text.
"rouge2": bigram (2-gram) based scoring - The model recalled X% of the two-word phrases from the reference text.
"rougeL": Longest common subsequence based scoring. - The model's longest sequence of words that matched the 
reference text covered X% of the reference text.
"rougeLSum": splits text using "\n" - The model's average longest common subsequence of words across sentences 
covered X% of the reference text.
"""
# ======================================================================================================================

# Define the evaluator
# To temporary store the results we will use cache_dir
rouge = evaluate.load("rouge", cache_dir="E:\\Repository\\Book\\models")

# get the score
dolly_result = rouge.compute(
    predictions=[dl_ans_1[0]["generated_text"]], references=[dolly_response_data]
)

print(dolly_result)
"""
Output:
-------
{'rouge1': 0.3835616438356165, 'rouge2': 0.08815426997245178, 'rougeL': 0.19178082191780824, 'rougeLsum': 0.2322946175637394}
"""

# get the score
dolly_result_2 = rouge.compute(
    predictions=[dl_ans_2[0]["generated_text"]], references=[dolly_response_data]
)

print(dolly_result_2)
"""
Output:
-------
{'rouge1': 0.35200000000000004, 'rouge2': 0.11678832116788321, 'rougeL': 0.3, 'rougeLsum': 0.3355704697986577}
"""

# Call eval on both input with their respective references.
dolly_result = rouge.compute(
    predictions=[dl_ans_1[0]["generated_text"], dl_ans_2[0]["generated_text"]],
    references=[dolly_response_data, dolly_response_data],
)
print(dolly_result)
"""
Output:
-------
{'rouge1': 0.36778082191780825, 'rouge2': 0.10247129557016749, 'rougeL': 0.24589041095890413, 'rougeLsum': 0.2839325436811985}
"""

# ======================================================================================================================
"""
BLEURT SCORE

BLEURT’s output is always a number. This value indicates how similar the generated text
is to the reference texts, with values closer to 1 representing more similar texts.
"""
# ======================================================================================================================
# Define the evaluator
# To temporary store the results we will use cache_dir
bleurt = evaluate.load("bleurt", cache_dir="E:\\Repository\\Book\\models")

bleurt_specific_data = " ".join([k for k in dolly_response_data])

# We can compute the eval matrix on multiple input with their respective reference as shown below.
# We can use it for any eval matrix not limited to this one like with the one above ROGUE score
bleurt_results = bleurt.compute(
    predictions=[dl_ans_1[0]["generated_text"], dl_ans_2[0]["generated_text"]],
    references=[bleurt_specific_data, bleurt_specific_data],
)

print(bleurt_results)
"""
Output:
-------
{'scores': [-1.241575002670288, -1.2617411613464355]}
"""

# ======================================================================================================================
"""
METEOR SCORE
Its values range from 0 to 1

HIGHER the score better the result
"""
# ======================================================================================================================
meteor = evaluate.load("meteor", cache_dir="E:\\Repository\\Book\\models")

mtr_results = meteor.compute(
    predictions=[dl_ans_1[0]["generated_text"]],
    references=[dolly_response_data],
)

print(mtr_results)
"""
Output:
-------
{'meteor': 0.32992160278745647}
"""


# ======================================================================================================================
"""
Perplexity SCORE
The Perplexity values are in the range of 0 to INFINITE.

LOWER the score better the result
"""
# ======================================================================================================================

# Define the evaluator
# To temporary store the results we will use cache_dir
perplexity = evaluate.load("perplexity", cache_dir="E:\\Repository\\Book\\models")

# model_id here we can not provide cache_dir hence it will be downloaded to default directory
# You will get this directory when you will run it
pxl_results = perplexity.compute(
    predictions=[dl_ans_2[0]["generated_text"]], model_id="databricks/dolly-v2-3b"
)

print(pxl_results)
"""
Output:
-------
{'perplexities': [6.705838203430176], 'mean_perplexity': 6.705838203430176}
"""
