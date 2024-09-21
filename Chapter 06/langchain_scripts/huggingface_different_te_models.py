"""
This script will demonstrate how to utilize opensource text embedding models.

https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending
"""

from langchain.document_loaders import WikipediaLoader
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

inference_api_key = "PUT_HUGGINGFACE_TOKEN_HERE"

text_to_embed = """
                    Text embedding models are like dictionaries for computers!
                    They turn words into numbers, capturing their meaning and how they relate to each other.
                    This lets computers understand the text and perform tasks like classifying emails,
                    searching for similar articles, or even translating languages.
                    Think of it as a secret code that unlocks the hidden insights within words.
                """

# ======================================================================================================================
# Let's see how to deal with text
# This is method 1
embeddings_model_1 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cpu"},  # For gpu replace cpu with cuda
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\models",
)

query_result_1 = embeddings_model_1.embed_query(text_to_embed)

# print generated vector embeddings
print(query_result_1)
# length of vec embedding
print(len(query_result_1))

"""
Output has been truncated
Output:
-------
[-0.0027904061134904623, -0.07718681544065475, 0.0003363988653291017, 0.030677713453769684, 0.030282968655228615,
.................................................................................................................
0.004473954439163208, -0.02310292050242424, 0.03343520686030388, 0.08505837619304657, -0.035957012325525284]
"""

# ......................................................................................................................
embeddings_model_2 = HuggingFaceEmbeddings(
    model_name="DataikuNLP/paraphrase-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # For gpu replace cpu with cuda
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\models",
)

query_result_2 = embeddings_model_2.embed_query(text_to_embed)

# print generated vector embeddings
print(query_result_2)
# length of vec embedding
print(len(query_result_2))

"""
Output has been truncated
Output:
-------
[-0.3654569983482361, -0.2156318575143814, -0.26118695735931396, -0.2503187358379364, 0.03771350905299187, 
........................................................................................................
0.5823591947555542, 0.08670958131551743, -0.1610865443944931, 0.53774094581604, -0.061369333416223526]
"""

# ......................................................................................................................
# Let's load the document from wikipedia and create vector embeddings of the same
# Here we are using one of the document loader
docs = WikipediaLoader(query="Large language model", load_max_docs=2).load()

# some details on the topic
print(len(docs))
[docs[k].metadata for k in range(0, 2)]

content_list = [docs[k].page_content for k in range(0, 2)]
print(len(content_list))

embeddings_model_3 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cpu"},  # For gpu replace cpu with cuda
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\models",
)

# embed_query won't work with list hence need to convert into string
query_result_3 = embeddings_model_3.embed_query(str(content_list))

# print generated vector embeddings
print(query_result_3)
# length of vec embedding
print(len(query_result_3))

"""
Output has been truncated
Output:
-------
[-0.00603552907705307, -0.10006360709667206, 0.009146483615040779, 0.003421128960326314, 0.013949036598205566,
..............................................................................................................
0.005309339612722397, 0.03647276759147644, 0.01297552790492773, -0.017824966460466385]
"""


# ======================================================================================================================
# Let's see how to deal with list of text/sentences
# You can use for plain text as well
# This is method 2

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
    "sentence-transformers/all-MiniLM-l6-v2",
    device="cpu",  # For gpu replace cpu with cuda
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
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

# ......................................................................................................................
# It will download the model of size around 100 MB
# The default path is ~/.cache/torch which can be overridden by cache_folder parameter
embeddings_model_5 = SentenceTransformer(
    "DataikuNLP/paraphrase-MiniLM-L6-v2",
    device="cpu",  # For gpu replace cpu with cuda
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
)

query_result_5 = embeddings_model_5.encode(text_to_embed)

# print generated vector embeddings
print(query_result_5)
# length of vec embedding
print(len(query_result_5))
# length of vec embedding of individual component
print(len(query_result_5[0]))

"""
Output has been truncated
Output:
-------
[[-0.7372107  -0.52178365 -0.25099593 ... -0.16200256  0.7495447
   0.00935555]
 [-0.37657952  0.29422578 -0.24300395 ... -0.12190361  0.6113903
  -0.19045316]
 [-0.66512805 -0.30456468 -0.09000997 ...  0.4875261   0.5887398
   0.01081237]
 [-0.47618088 -0.00236685 -0.5388156  ...  0.17080715  0.09239917
  -0.13250606]
 [-0.23935585 -0.33497378 -0.28933358 ...  0.17934461  0.43651223
  -0.35096776]]
"""

# ......................................................................................................................
# Let's load the document from wikipedia and create vector embeddings of the same
# Here we are using one of the document loader
docs = WikipediaLoader(query="Large language model", load_max_docs=2).load()

# some details on the topic
print(len(docs))
[docs[k].metadata for k in range(0, 2)]

content_list = [docs[k].page_content for k in range(0, 2)]
print(len(content_list))

embeddings_model_6 = SentenceTransformer(
    "sentence-transformers/all-MiniLM-l6-v2",
    device="cpu",  # For gpu replace cpu with cuda
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
)

query_result_6 = embeddings_model_6.encode(content_list)

# print generated vector embeddings
print(query_result_6)
# length of vec embedding
print(len(query_result_6))
# length of vec embedding of individual component
print(len(query_result_6[0]))

"""
Output has been truncated
Output:
-------
[[-2.31653568e-03 -9.77388844e-02 -5.47833880e-03  1.66091267e-02
  ...............................................................
  -1.58348666e-05  1.17695238e-02  9.09951888e-03 -1.54658202e-02]
 [-8.48497450e-02 -1.09056398e-01  3.93328331e-02  2.19532009e-02
   ..............................................................
   8.51376913e-03  2.77478900e-02  1.70640890e-02 -5.86922541e-02]]
"""
