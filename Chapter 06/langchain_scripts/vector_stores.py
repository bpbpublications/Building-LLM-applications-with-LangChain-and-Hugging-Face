"""
This script demonstrate usage of vector store.
Here we will see 2 vector store Chromadb and Faiss

https://python.langchain.com/docs/integrations/vectorstores/chroma
https://python.langchain.com/docs/integrations/vectorstores/faiss
"""

from langchain.document_loaders import WikipediaLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS

inference_api_key = "PUT_HUGGINGFACE_TOKEN_HERE"

# Let's load the document from wikipedia and create vector embeddings of the same
# Here we are using one of the document loader
docs = WikipediaLoader(query="Large language model", load_max_docs=10).load()

# some details on the topic
print(len(docs))
[docs[k].metadata for k in range(0, 10)]
[docs[k].page_content for k in range(0, 10)]

embeddings_model_6 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
)


# ======================================================================================================================
# USING CHROMADB
# ======================================================================================================================

# save to disk
db1 = Chroma.from_documents(
    docs, embeddings_model_6, persist_directory="E:\\Repository\\Book\\chroma_db"
)

# now ask the questions
# The function .similarity_search will return k number of documents most similar to the query.
# Default value for k is 4 which means it returns 4 similar documents.
# To override the behavior mention k=1 or k=2 to return only 1 or 2 similar documents.
qa1 = db1.similarity_search("What is training cost?")

# print all similar docs
print(qa1)

# print first doc, same way replace 0 with 1 to 3 numbers to get remaining 3 docs content
print(qa1[0].page_content)

# ======================================================================================================================
# We can create another function where we will load saved vec embedding and use it further.
# Below we will see how to do that

# First import the packages

# Define model
embeddings_model_6 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
)

# load saved vec embedding from disk
db2 = Chroma(
    persist_directory="E:\\Repository\\Book\\chroma_db",
    embedding_function=embeddings_model_6,
)

# ask question
# The function .similarity_search will return k number of documents most similar to the query.
# Default value for k is 4 which means it returns 4 similar documents.
# To override the behavior mention k=1 or k=2 to return only 1 or 2 similar documents.
qa2 = db2.similarity_search(
    "Explain Large Language Models in funny way so that child can understand."
)

# print all similar docs
print(qa2)

# print first doc, same way replace 0 with 1 to 3 numbers to get remaining 3 docs content
print(qa2[0].page_content)


# ======================================================================================================================
# USING FAISS
# ======================================================================================================================

# save to disk
db3 = FAISS.from_documents(docs, embeddings_model_6)
# For FAISS single slash in path has not worked hence need to give the double slash
db3.save_local(folder_path="E:\\Repository\\Book\\faiss_db")

# now ask the questions
# The function .similarity_search will return k number of documents most similar to the query.
# Default value for k is 4 which means it returns 4 similar documents.
# To override the behavior mention k=1 or k=2 to return only 1 or 2 similar documents.
qa3 = db3.similarity_search("What is training cost?")

# print all similar docs
print(len(qa3))
print(qa3)

# print 3rd doc, same way replace 3 with 0,1,2 numbers to get remaining 3 docs content
print(qa3[3].page_content)

# ======================================================================================================================
# We can create another function where we will load saved vec embedding and use it further.
# Below we will see how to do that

# First import the packages

# Define model
embeddings_model_6 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
)

# load saved vec embedding from disk
db4 = FAISS.load_local(
    folder_path="E:\\Repository\\Book\\faiss_db",
    embeddings=embeddings_model_6,
    # ValueError: The de-serialization relies loading a pickle file. Pickle files can be modified to deliver
    # a malicious payload that results in execution of arbitrary code on your machine.You will need to set
    # `allow_dangerous_deserialization` to `True` to enable deserialization. If you do this, make sure that
    # you trust the source of the data. For example, if you are loading a file that you created, and know that
    # no one else has modified the file, then this is safe to do. Do not set this to `True` if you are loading
    # a file from an untrusted source (e.g., some random site on the internet.).
    allow_dangerous_deserialization=True,
)

# ask question
# The function .similarity_search will return k number of documents most similar to the query.
# Default value for k is 4 which means it returns 4 similar documents.
# To override the behavior mention k=1 or k=2 to return only 1 or 2 similar documents.
qa4 = db4.similarity_search(
    "Explain Large Language Models in funny way so that child can understand."
)

# print all similar docs
print(qa4)

# print 2nd doc, same way replace 2 with 0, 1, 3 numbers to get remaining 3 docs content
print(qa4[2].page_content)
