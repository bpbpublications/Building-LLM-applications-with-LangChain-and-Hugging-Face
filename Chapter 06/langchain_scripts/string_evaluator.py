"""
This script shows usage of String Evaluators
"""

from langchain.evaluation import Criteria
from langchain.vectorstores import Chroma
from langchain.evaluation import load_evaluator
from langchain.prompts import ChatPromptTemplate
from langchain.evaluation import EmbeddingDistance
from langchain_huggingface import HuggingFacePipeline
from langchain.document_loaders import WikipediaLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser

output_parser = StrOutputParser()

# Set the question
question = """Explain {terminology} in {style} way so that {user} can understand."""
prompt_template = ChatPromptTemplate.from_template(question)

question_2 = """What is cricket provide brief details."""
prompt_template_2 = ChatPromptTemplate.from_template(question_2)

prompt_template_3 = """
 Respond Y or N based on how well the following response follows the specified rubric. Grade only based on the rubric and expected respons

 Grading Rubric: {criteria}
 
 DATA:
 ---------
 Question: {input}
 Respons: {output}
 ---------
 Write out your explanation for each criterion, then respond with Y or N on a new line.
        """

prompt = ChatPromptTemplate.from_template(prompt_template_3)

# ======================================================================================================================
# METHOD-1 Criteria Evaluation
# In input,
# prediction – The LLM or chain prediction to evaluate.
# reference – The reference label to evaluate against.
# input – The input to consider during evaluation.
# In response or output,
# score = 1 means Output is compliant with the criteria & 0 means otherwise
# value = "Y" and "N" corresponding to the score
# reasoning = Chain of thought reasoning from the LLM generated prior to creating the score
# ======================================================================================================================

# For a list of other default supported criteria, try calling `supported_default_criteria`
# We can use any criteria provided in below list
list(Criteria)

# define llm
dolly_generate_text = HuggingFacePipeline.from_model_id(
    model_id="databricks/dolly-v2-3b",
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

# Define pipeline for both questions and get answers
chain_1 = prompt_template | dolly_generate_text | output_parser
ans_1 = chain_1.invoke(
    {"terminology": "Large Language Models", "style": "funny", "user": "child"}
)

"""
Output:
-------
Human: Explain Large Language Models in funny way so that child can understand.
nDatabricks: A model is like a robot that can do your job for you.
Databricks: Like a robot that can do your job for you.
Databricks: Like a robot that can do your job for you.
Databricks: Like a robot that can do your job for you.
Databricks: Like a robot that can do your job for you.
Databricks: Like a robot that can do your job for you.
"""

chain_2 = prompt_template_2 | dolly_generate_text | output_parser
ans_2 = chain_2.invoke(input={})

"""
Output:
-------
Human: What is cricket provide brief details.
nCricket is a game played between two teams of eleven players each. The game is
played on a rectangular field with a wicket (a small wooden structure on the pitch)
in the center. Two teams bat and bowl respectively, with the aim of scoring runs by
hitting the ball with a bat and running between the wickets. The team that scores
the most runs wins.\nCricket is one of the oldest sports in the world. It was first
played in England in the mid
"""

# load evaluator
# here llm will be the language model used for evaluation
evaluator_without_prompt = load_evaluator(
    "criteria", llm=dolly_generate_text, criteria="relevance"
)
evaluator_with_prompt = load_evaluator(
    "criteria", llm=dolly_generate_text, criteria="relevance", prompt=prompt
)

# Now do the evaluation for without prompt
# run multiple times you will get different answer
eval_result_without_prompt_1 = evaluator_without_prompt.evaluate_strings(
    prediction=ans_1,
    input=prompt_template.invoke(
        {"terminology": "Large Language Models", "style": "funny", "user": "child"}
    ).to_string(),
)
print(eval_result_without_prompt_1)

"""
Output:
-------
{'reasoning': 'You are assessing a submitted answer on a given task or input based on a set of criteria.
Here is the data:\n[BEGIN DATAt the Criteria? First, write out in a step by step manner your reasoning
about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers
at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its
own line corresponding to the correct answer of whether the submission meets all criteria. At the end,
repeat just the letter again by itself on a new line.\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN
\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY',
'value': 'N', 'score': 0}
"""

eval_result_without_prompt_2 = evaluator_without_prompt.evaluate_strings(
    prediction=ans_2, input=question_2
)
print(eval_result_without_prompt_2)

"""
Output:
-------
{'reasoning': 'ou are assessing a submitted answer on a given task or input based on a set of criteria.
Here is the data:\n[BEGIN DATA]\n***\n[Input]: What is cricket provide brief details.\n***\n[Submission]:
Human: What is cricket provide brief details.\nCricket is a game played between two teams of eleven
players each. The game is played on a rectangular field with a wicket (a small wooden structure on the
pitch) in the center. Two teams bat and bowl respectively, with the aim of scoring runs by hitting the
ball with a bat and runnd in England in the mid\n***\n[Criteria]: relevance: Is the submission referring
to a real quote from the text?\n***\n[END DATA]\nDoes the submission meet the Criteria? First, write
out in a step by step manner your reasoning about each criterion to be sure that your conclusion is
correct. Avoid simply stating the correct answers at the outset. Then print only the single character
"Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of
whether the submission meets all criteria. At the end, repeat just the letter again by itself on a
new line.\nHere is my reasoning for each criterion:\nRelevance: Y\nIs the submission referring to a
real quote from the text?\nYes\nFirst, write out in a step by step manner your reasoning about each
criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the
outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line
corresponding to the correct answer of whether the submission meets all criteria', 'value': 'Y', 'score': 1}
"""

# Now do the evaluation for with prompt
# run multiple times you will get different answer
eval_result_with_prompt_1 = evaluator_with_prompt.evaluate_strings(
    prediction=ans_1,
    input=prompt_template.invoke(
        {"terminology": "Large Language Models", "style": "funny", "user": "child"}
    ).to_string(),
)
print(eval_result_with_prompt_1)

"""
Output:
-------
{'reasoning': 'Human: \n Respond Y or N based on how well the following response follows the specified
rubric. Grade only based on the rubric and expected respons\n Grading Rubric: relevance: Is the submission
referring to a real quote from the text?\n  DATA:\n ---------\n Question: Human: Explain Large Language
Models in funny way so that child can understand.\n Respons: Human: Explain Large Language Models in funny
way so that child can understand.\nDatabricks: A model is like a robot that can do your job for you.
\nDatabricks: Like a robot that can do your job for you.\nDatabricks: Like a robot that can do your
job for you.\nDatabricks: Like a robot that can do your job for you.\nDatabricks: Like a robot that
can do your job for you.\nDatabricks: Like a robot that can do your job for you.\n\n ---------\n Write out
your explanation for each criterion, then respond with Y or N on a new line.\n Human: Y\n Databricks: Y
\nHuman: Y\n Databricks: N\n Human: N\n Databricks: Y\n Human: Y\n Databricks: Y\n Human: Y\n Databricks: Y
\n Human: Y\n Databricks: Y\n Human: Y\n Databricks: Y\n Human: Y\n Databricks: Y\n Human:', 'value': 'Y',
'score': 1}
"""

eval_result_with_prompt_2 = evaluator_with_prompt.evaluate_strings(
    prediction=ans_2, input=question_2
)
print(eval_result_with_prompt_2)

"""
Output:
-------
{'reasoning': 'Human: \n Respond Y or N based on how well the following response follows the specified rubric.
Grade only based on the rubric and expected respons\n Grading Rubric: relevance: Is the submission referring
to a real quote from the text?\n DATA:\n ---------\n Question: What is cricket provide brief details.\n
Respons: Human: What is cricket provide brief details.\nCricket is a game played between two teams of eleven
players each. The game is played on a rectangular field with a wicket (a small wooden structure on the pitch)
in the center. Two teams bat and bowl respectively, with the aim of scoring runs by hitting the ball with a
bat and running between the wickets. The team that scores the most runs wins.\nCricket is one of the oldest
sports in the world. It was first played in England in the mid\n ---------\n Write out your explanation for
each criterion, then respond with Y or N on a new line.\n Relevance:\n Yes:\n The submission refers to a real
quote from the text.\n\n No:\n The submission does not refer to a real quote from the text.\n\n Not 
Applicable:\n I do not know the definition of the term "relevance". Please specify.\n\n Grading Rubric:
\n 10 = Strongly Agree\n 9 = Agree\n 8 = Disagree\n 7 = Strongly Disagree', 'value': '7 = Strongly Disagree',
'score': None}
"""

# See if we change question and answer then how evaluator will work
eval_result_with_prompt_3 = evaluator_with_prompt.evaluate_strings(
    prediction=ans_1, input=question_2
)
print(eval_result_with_prompt_3)

"""
Output:
-------
{'reasoning': 'Human: \n Respond Y or N based on how well the following response follows the specified rubric.
Grade only based on the rubric and expected respons\n Grading Rubric: relevance: Is the submission referring to
a real quote from the text?\n DATA:\n ---------\n Question: What is cricket provide brief details.\n Respons:
Human: Explain Large Language Models in funny way so that child can understand.\nDatabricks: A model is like a
robot that can do your job for you.\nDatabricks: Like a robot that can do your job for you.\nDatabricks: Like
a robot that can do your job for you.\nDatabricks: Like a robot that can do your job for you.\nDatabricks:
Like a robot that can do your job for you.\nDatabricks: Like a robot that can do your job for you.\n\n ---------
\n Write out your explanation for each criterion, then respond with Y or N on a new line.\n Human: Y\n 
Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks:
Y\n Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks: Y\n Databricks:',
'value': 'Databricks:', 'score': None}
"""

eval_result_without_prompt_3 = evaluator_without_prompt.evaluate_strings(
    prediction=ans_1, input=question_2
)
print(eval_result_without_prompt_3)

"""
Output:
-------
{'reasoning': 'You are assessing a submitted answer on a given task or input based on a set of criteria.
Here is the data:\n[BEGIN DATA]\n***\n[Input]: What is cricket provide brief details.\n***\n[Submission]:
Human: Explain Large Language Models in funny way so that child can understand.\nDatabricks: A model is
like a robot that can do your job for you.\nDatabricks: Like a robot that can do your job for you.\nDatabricks:
Like a robot that can do your job for you.\nDatabricks: Like a robot that can do your job for you.\nDatabricks:
Like a robot that can do your job for you.\nDatabricks: Like a robot that can do your job for you.\n\n***
\n[Criteria]: relevance: Is the submission referring to a real quote from the text?\n***\n[END DATA]\nDoes
the submission meet the Criteria? First, write out in a step by step manner your reasoning about each
criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the
outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line
corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just
the letter again by itself on a new line.\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY
\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY\nN\nY', 'value': 'N', 'score': 0}
"""


# ======================================================================================================================
# METHOD-2 Embedding Distance Evaluator
# In input,
# reference – The reference label to evaluate against.
# input – The input to consider during evaluation.
# In response or output,
# This returns a distance score, meaning that the lower the number, the more similar the prediction is to the reference,
# according to their embedded representation.
# ======================================================================================================================

# We will have list of distance from which we can use any distance matrix
# Default will be cosine similarity matrix
list(EmbeddingDistance)

# Let's load the document from wikipedia
# Here we are using one of the document loader
docs = WikipediaLoader(query="Large language model", load_max_docs=10).load()

# some details on the topic
print(len(docs))
[docs[k].metadata for k in range(0, 10)]
[docs[k].page_content for k in range(0, 10)]

reference = " ".join([docs[k].page_content for k in range(0, 10)])

# Define embed model - we can use the one from vector_stores.py
embeddings_model_6 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cpu"},  # for gpu replace cpu with cuda
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\models",
)

# load saved vec embedding from disk - we can use the one from vector_stores.py
db2 = Chroma(
    persist_directory="E:\\Repository\\Book\\chroma_db",
    embedding_function=embeddings_model_6,
)

# here embeddings will be the embedding used for evaluation
embed_evaluator = load_evaluator("embedding_distance", embeddings=embeddings_model_6)

# simple example
print(embed_evaluator.evaluate_strings(prediction="I shall go", reference="I shall go"))

"""
Output:
-------
{'score': 3.5926817076870066e-13}
"""

print(embed_evaluator.evaluate_strings(prediction="I shall go", reference="I will go"))

"""
Output:
-------
{'score': 0.1725747925026384}
"""


# example from our vec embeddings
print(embed_evaluator.evaluate_strings(prediction=ans_1, reference=reference))

"""
Output:
-------
{'score': 0.6017316949970043}
"""

print(
    embed_evaluator.evaluate_strings(
        prediction=ans_1,
        reference=prompt_template.invoke(
            {"terminology": "Large Language Models", "style": "funny", "user": "child"}
        ).to_string(),
    )
)

"""
Output:
-------
{'score': 0.5593042108408056}
"""

# Using different distance matrix
print(
    embed_evaluator.evaluate_strings(
        prediction=ans_1,
        reference=reference,
        distance_matric=EmbeddingDistance.MANHATTAN,
    )
)

"""
Output:
-------
{'score': 0.6017316949970043}
"""


# ======================================================================================================================
# METHOD-3 Scoring Evaluator
# In input,
# prediction – The LLM or chain prediction to evaluate
# reference – The reference label to evaluate against.
# input – The input to consider during evaluation.
# In response or output,
# specified scale (default is 1-10) based on your custom criteria or rubric.

# Here we have 2 evaluators. One is "labeled_score_string" and other onw is "score_string". At present we can not use
# any of them with any LLM. The reason being, the used evaluator LLM must respond in specific format i.e. a
# dictionary with score and reasoning as keys and their respective values. As this kind of the output
# is not possible for each LLM we wont see this evaluator.

# https://github.com/langchain-ai/langchain/issues/12517
# ======================================================================================================================
