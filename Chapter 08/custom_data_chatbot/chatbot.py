"""
The script will create play ground to test chatbot
"""

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# ======================================================================================================================
# Defining global settings for easy and fast work

# load text embedding model from HuggingFaceHub to generate vector embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cpu"},  # for gpu replace cpu with cuda
    encode_kwargs={"normalize_embeddings": False},
    cache_folder="E:\\Repository\\Book\\models",
    multi_process=False,
)

chroma_db = Chroma(
    persist_directory="E:\\Repository\\Book\\chroma_db", embedding_function=embed_model
)


# Retrieve .............................................................................................................
# define retriever to retrieve Question related Docs
retriever = chroma_db.as_retriever(
    search_type="mmr",  # Maximum MArginal Relevance
    search_kwargs={"k": 8},  # max relevan docs to retrieve
)


dolly_generate_text = pipeline(
    model="databricks/dolly-v2-3b",
    token="PUT_HUGGINGFACEHUB_TOKEN_HERE",
    trust_remote_code=True,
    device_map="auto",  # make it "auto" for auto selection between GPU and CPU, -1 for CPU, 0 for GPU
    return_full_text=True,  # necessary to return complete text.
    tokenizer=AutoTokenizer.from_pretrained("databricks/dolly-v2-3b"),
    temperature=0.1,  # to reduce randomness in the answer
    max_new_tokens=1000,  # generate this number of tokens
    # change the cache_dir based on your preferences
    # model kwargs are for model initialization
    model_kwargs={
        "cache_dir": "E:\\Repository\\Book\\models",
        "offload_folder": "offload",  # use it when model size is > 7B
    },
)

dolly_pipeline_hf = HuggingFacePipeline(pipeline=dolly_generate_text)

retrievalQA = RetrievalQA.from_llm(llm=dolly_pipeline_hf, retriever=retriever)


def chatbot(input_text: str) -> str:
    """
    This function will provide the answer of the queries. Here first we will load the stored

    Parameters
    ----------

    input_text: str
        User's question

    """

    ans = retrievalQA.invoke(input=input_text)
    return ans["result"]


iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Information Retrieval Bot",
)


iface.launch(share=True)
