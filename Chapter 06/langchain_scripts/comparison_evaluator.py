"""
This script shows usage of String Evaluators
"""

import os
from getpass import getpass
from langchain.evaluation import load_evaluator
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint

output_parser = StrOutputParser()

# Prompt to put token. When requested put the token that we have generated
HUGGINGFACEHUB_API_TOKEN = getpass()

# Set the environment variable to use the token locally
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Set the question
question = """Explain {terminology} in {style} way so that {user} can understand."""
prompt_template = ChatPromptTemplate.from_template(question)

question_2 = """What is cricket provide brief details."""
prompt_template_2 = ChatPromptTemplate.from_template(question_2)

# define first llm and its responses------------------------------------------------------------------------------------
# These calls are online call i.e. calling API
falcon_llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b",
    # Based on the requirement we can change the values. Bases on the values time can vary
    temperature=0.5,
    do_sample=True,
    timeout=300,
)

# Define pipeline for both questions and get answers
chain_1 = prompt_template | falcon_llm | output_parser
ans_11 = chain_1.invoke(
    {"terminology": "Large Language Models", "style": "funny", "user": "child"}
)

chain_2 = prompt_template_2 | falcon_llm | output_parser
ans_12 = chain_2.invoke(input={})

# define second llm and its responses-----------------------------------------------------------------------------------
# These calls are online call i.e. calling API
ms_llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    # Based on the requirement we can change the values. Bases on the values time can vary
    temperature=0.5,
    do_sample=True,
    timeout=300,
)

# Define pipeline for both questions and get answers
chain_3 = prompt_template | ms_llm | output_parser
ans_21 = chain_3.invoke(
    {"terminology": "Large Language Models", "style": "funny", "user": "child"}
)

chain_4 = prompt_template_2 | ms_llm | output_parser
ans_22 = chain_4.invoke(input={})

# Let's load the document from wikipedia
# Here we are using one of the document loader
docs = WikipediaLoader(query="Large language model", load_max_docs=10).load()

# some details on the topic
print(len(docs))
[docs[k].metadata for k in range(0, 10)]
[docs[k].page_content for k in range(0, 10)]

reference = " ".join([docs[k].page_content for k in range(0, 10)])


# ======================================================================================================================
# METHOD-1 Pairwise String Comparison
# In input,
# prediction – The LLM or chain prediction to evaluate.
# reference – The reference label to evaluate against.
# input – The input to consider during evaluation.
# In response or output,
# score = 1 means Output is compliant with the criteria & 0 means otherwise
# value = "Y" and "N" corresponding to the score
# reasoning = Chain of thought reasoning from the LLM generated prior to creating the score
# ======================================================================================================================

# In online llm i.e. via API call we might get timeout or any other issue hence we will define local llm
ms_generate_text = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    device_map="auto",  # Automatically distributes the model across available GPUs and CPUs
    # Based on the requirement we can change the values. Bases on the values time can vary
    pipeline_kwargs={
        "max_new_tokens": 100,  # generate maximum 100 new tokens in the output
        "do_sample": False,  # Less diverse and less creative answer.
        "repetition_penalty": 1.03,  # discourage from generating repetative text
    },
    model_kwargs={
        "cache_dir": "E:\\Repository\\Book\\models",  # store data into give directory
        "offload_folder": "offload",
    },
)

# string_evaluator = load_evaluator("labeled_pairwise_string", llm=falcon_llm) # In case we have reference available
# string_evaluator_1 = load_evaluator("pairwise_string", llm=falcon_llm) # In case reference is not available

# In case above llm via API call gives any kind of the error we can use locally defined llm
string_evaluator = load_evaluator(
    "labeled_pairwise_string", llm=ms_generate_text
)  # In case we have reference available
string_evaluator_1 = load_evaluator(
    "pairwise_string", llm=ms_generate_text
)  # In case reference is not available

# It will take too much time
string_evaluator.evaluate_string_pairs(
    prediction=ans_11,
    prediction_b=ans_21,
    input=prompt_template.invoke(
        {"terminology": "Large Language Models", "style": "funny", "user": "child"}
    ).to_string(),
    reference=reference,
)

string_evaluator_1.evaluate_string_pairs(
    prediction=ans_11,
    prediction_b=ans_21,
    input=prompt_template.invoke(
        {"terminology": "Large Language Models", "style": "funny", "user": "child"}
    ).to_string(),
)

string_evaluator_1.evaluate_string_pairs(
    prediction=ans_12,
    prediction_b=ans_22,
    input=prompt_template_2.invoke(input={}).to_string(),
)

# ======================================================================================================================
"""
If above does not work do not worry. It seems that its right now working with OpenAI based LLMs and not with other 
LLMs. The reason being, the used evaluator LLM must respond in specific format and as the specific format is not 
possible for each LLM we wont see this evaluator. It will raise an error Output must contain a double bracketed string
with the verdict 'A', 'B', or 'C'.

https://github.com/langchain-ai/langchain/issues/12517
"""
# ======================================================================================================================
