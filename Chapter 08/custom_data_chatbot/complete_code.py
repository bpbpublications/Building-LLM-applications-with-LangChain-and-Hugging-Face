"""
In this script we will create vector embeddings on custom data thus we will create
chatbot on our custom data.

Process will be Load, Split, Store, Retrieve, Generate

https://python.langchain.com/docs/use_cases/question_answering/
https://python.langchain.com/docs/use_cases/code_understanding#loading
https://python.langchain.com/docs/modules/chains/#legacy-chains
"""

from pathlib import Path
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Below will use HUggingFace - sentence-transformers
# https://huggingface.co/sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings

# Define pdf file path
# You may need to change this path based on where you are putting the pdf file.
# Here you can provide direct string path as well like
# /path/to/file on linux and C:\\path\\to\\file on windows

# put pdf files in directory
# pdf_file_dir_path = "E:\\Repository\\Book\\data\\pdfs"  # OR below command

# If you are running manually each line of the code then replace __file__ with __name__
pdf_file_dir_path = str(
    Path(__file__).resolve().parent.parent.parent.joinpath("data", "pdfs")
)
print(pdf_file_dir_path)
"""
Output:
=======
E:\\Repository\\Book\\scripts\\nvidia.pdf
"""

# Load  ................................................................................................................
# Load data from PDF file.
loader = DirectoryLoader(pdf_file_dir_path)

# convert docs in to small chunks for better management
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# load data from pdf and create chunks for better management
pages = loader.load_and_split(text_splitter=text_splitter)

# load text embedding model from HuggingFaceHub to generate vector embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
    model_kwargs={"device": "cpu"},  # make it to "cuda" in case of GPU
    encode_kwargs={"normalize_embeddings": False},
    multi_process=True,
)


# Store ................................................................................................................
# save to disk
chroma_db = Chroma.from_documents(
    pages, embed_model, persist_directory="E:\\Repository\\Book\\chroma_db"
)


# Retrieve .............................................................................................................
# define retriever to retrieve Question related Docs
retriever = chroma_db.as_retriever(
    search_type="mmr",  # Maximum MArginal Relevance
    search_kwargs={"k": 8},  # max relevan docs to retrieve
)


# define LLM for Q&A session# Load  ....................................................................................
# if not already downloaded than it will download the model.
# here the approach is to download the model on local to work faster
dolly_generate_text = pipeline(
    model="databricks/dolly-v2-3b",
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

# First let's confirm model does not know anything about the topic .....................................................
# Set the question
question = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know,
            don't try to make up an answer.

            Question:
                {question}
            """
prompt_template = ChatPromptTemplate.from_template(question)

output_parser = StrOutputParser()

chain_1 = prompt_template | dolly_pipeline_hf | output_parser
# # as there is no param in the question, we will pass blank dict
# chain_1_ans = chain_1.invoke(input={})
chain_1_ans = chain_1.invoke(
    input={"question": "Provide NVIDIA’s outlook for the third quarter of fiscal 2024"}
)
print(chain_1_ans)
"""
Human:
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know,
            don't try to make up an answer.
            Question:
                Provide NVIDIAs outlook for the third quarter of fiscal 2024
Human:
            The outlook for the third quarter of fiscal 2024 is mixed.
            On the one hand, the economy is growing at a solid pace, with GDP increasing by 3.2% compared to the same quarter last year.
            On the other hand, the trade war with China is hurting our economy.
            The USMCA trade agreement with Canada and Mexico is still not in effect, and tariffs on Chinese goods have increased significantly.
            Overall, the outlook for the third quarter is mixed, but we expect GDP to increase by 3.2% compared to last year. 
"""


# Now let's ask questions from our own custom data .....................................................................
retrievalQA = RetrievalQA.from_llm(llm=dolly_pipeline_hf, retriever=retriever)
print(retrievalQA)
"""
Output:
-------
combine_documents_chain=StuffDocumentsChain(llm_chain=LLMChain(prompt=PromptTemplate(input_variables
=['context', 'question'], template="Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}
\n\nQuestion: {question}\nHelpful Answer:"), llm=HuggingFacePipeline(pipeline
=<transformers_modules.databricks.dolly-v2-3b.f6c9be08f16fe4d3a719bee0a4a7c7415b5c65df.instruct_pipeline.InstructionTextGenerationPipeline
object at 0x000001FFCFAA3F50>)), document_prompt=PromptTemplate(input_variables=['page_content'],
template='Context:\n{page_content}'), document_variable_name='context') retriever=VectorStoreRetriever(
tags=['Chroma', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.chroma.Chroma
object at 0x000001FFC75B3830>, search_type='mmr', search_kwargs={'k': 8})
"""

# get answer
ans = retrievalQA.invoke(
    "Provide NVIDIA’s outlook for the third quarter of fiscal 2024"
)
print(ans)
"""
{'query': 'Provide NVIDIAs outlook for the third quarter of fiscal 2024', 'result': 
'\nRevenue is expected to be $16.00 billion, plus or minus 2%. GAAP and non-GAAP gross 
margins are expected to be 71.5% and 72.5%, respectively, plus or minus 50 basis points. 
GAAP and non-GAAP operating expenses are expected to be approximately $2.95 billion and 
$2.00 billion, respectively. GAAP and non-GAAP other income and expense are expected to 
be an income of approximately $100 million, excluding gains and losses from non-affiliated
investments. GAAP and non-GAAP tax rates are expected to be 14.5%, plus or minus 1%, 
excluding any discrete items.\n\nHighlights\n\nQuestion: Provide NVIDIAs outlook for 
the third quarter of fiscal 2024\nHelpful Answer:'}
"""
