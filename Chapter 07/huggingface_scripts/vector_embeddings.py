"""
This script will demonstrate how to create vector embedding using sentence_transformers package.
https://huggingface.co/docs/hub/sentence-transformers
https://huggingface.co/sentence-transformers
https://www.sbert.net/

Please note that for publicly available models the token is not required.
"""

from sentence_transformers import SentenceTransformer

token = "PUT_HUGGINGFACE_TOKEN_HERE"

text_to_embed = """
                    Text embedding models are like dictionaries for computers!
                    They turn words into numbers, capturing their meaning and how they relate to each other.
                    This lets computers understand the text and perform tasks like classifying emails,
                    searching for similar articles, or even translating languages.
                    Think of it as a secret code that unlocks the hidden insights within words.
                """

# ======================================================================================================================
# Let's see how to deal with text
embeddings_model_1 = SentenceTransformer(
    model_name_or_path="sentence-transformers/all-MiniLM-l6-v2",
    token=token,
    device="cpu",  # for gpu replace cpu with cuda
    cache_folder="E:\\Repository\\Book\\models",
)

query_result_1 = embeddings_model_1.encode(text_to_embed)

# print generated vector embeddings
print(query_result_1)
# length of vec embedding
print(len(query_result_1))

"""
Output has been truncated
Output:
-------
[-2.79038935e-03 -7.71868527e-02  3.36391415e-04  3.06777228e-02
  ..............................................................
 -2.31029969e-02  3.34352329e-02  8.50583911e-02 -3.59569825e-02]
"""
# ======================================================================================================================
# Let's see how to deal with list of text/sentences

text_to_embed = [
    "Text embedding models are like dictionaries for computers!",
    "They turn words into numbers, capturing their meaning and how they relate to each other.",
    "This lets computers understand the text and perform tasks like classifying emails, searching for similar articles,"
    "or even translating languages.",
    "Think of it as a secret code that unlocks the hidden insights within words.",
    "A large language model, like GPT-3.5, leverages vast datasets to understand and generate human-like text across"
    "diverse subjects.",
]

print(len(text_to_embed))

# ......................................................................................................................
# It will download the model of size around 100 MB
# The default path is ~/.cache/torch which can be overridden by cache_folder parameter
embeddings_model_4 = SentenceTransformer(
    model_name_or_path="sentence-transformers/all-MiniLM-l6-v2",
    token=token,
    device="cpu",  # for gpu replace cpu with cuda
    cache_folder="E:\\Repository\\Book\\models",
)

query_result_4 = embeddings_model_4.encode(text_to_embed)

# print generated vector embeddings
print(query_result_4)
# length of vec embedding
print(len(query_result_4))
# length of vec embedding of individual component
print(len(query_result_4[0]))

"""
Output has been truncated
Output:
-------
[[ 0.00476223 -0.08366839  0.02533819 ...  0.0081036   0.08216282
   0.00848225]
 [ 0.02075923  0.02187491 -0.04436149 ...  0.04193671  0.10981567
  -0.05544527]
 [-0.05549927  0.02617585 -0.05102286 ...  0.09186588  0.04069077
  -0.01355496]
 [-0.09845991  0.02013757 -0.05561479 ...  0.05502703  0.02024567
  -0.05868284]
 [-0.04475563 -0.07107755  0.02242337 ...  0.07566341  0.00079719
  -0.0443915 ]]
"""
