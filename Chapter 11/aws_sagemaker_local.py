"""
We are providing the code here which you can paste as is in Jupyter Notebook.
You can paste the code in single cell or based on the headings you can put it in different sections.

If any time you face error related to storage space is full run following commands
from notebook which will free up the space.

# !sudo rm -rf /tmp/*
# !sudo rm -rf /home/ec2-user/.cache/huggingface/hub/*
# !sudo rm -rf custom_data_chatbot/models/*
# !sudo rm -rf /home/ec2-user/SageMaker/.Trash-1000/*
"""

# import packages ......................................................................................................
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

# Below will use HuggingFace - sentence-transformers
# https://huggingface.co/sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings


# Define directories
pdf_file_dir_path = "custom_data_chatbot/pdfs"
model_path = "custom_data_chatbot/models"


# Load  ................................................................................................................
# Load data from PDF file.
loader = DirectoryLoader(pdf_file_dir_path)

# convert docs in to small chunks for better management
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

# load data from pdf and create chunks for better management
pages = loader.load_and_split(text_splitter=text_splitter)


# load text embedding model from HuggingFaceHub to generate vector embeddings ..........................................
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    cache_folder=model_path,
    # cpu because on AWS we are not using GPU
    model_kwargs={
        "device": "cpu",
    },  # make it to "cpu" in case of no GPU
    encode_kwargs={"normalize_embeddings": False},
    multi_process=True,
)


# Store vector embeddings and define retriever .........................................................................
chroma_db = Chroma.from_documents(pages, embed_model, persist_directory=model_path)

retriever = chroma_db.as_retriever(
    search_type="mmr",  # Maximum MArginal Relevance
    search_kwargs={"k": 1},  # max relevan docs to retrieve
)


# Load the pre-trained model and tokenizer .............................................................................
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=model_path)
model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=model_path)


# Define pipeline ......................................................................................................
text_generator = pipeline(
    task="text-generation",
    model=model,
    token="PUT_HERE_HUGGINGFACEHUB_API_TOKEN",
    trust_remote_code=True,
    device_map="auto",  # make it "auto" for auto selection between GPU and CPU, -1 for CPU, 0 for GPU
    tokenizer=tokenizer,
    max_length=1024,  # generate token sequences of 1024 including input and output token sequences
)

ms_dialo_gpt_hf = HuggingFacePipeline(pipeline=text_generator)


# Get Answer ...........................................................................................................
retrievalQA = RetrievalQA.from_llm(
    llm=ms_dialo_gpt_hf,
    retriever=retriever,
    prompt=PromptTemplate(
        input_variables=["context"],
        template="{context}",
    ),
)
print(retrievalQA)


# get answer
retrievalQA.invoke("Provide NVIDIA’s outlook for the third quarter of fiscal 2024")

"""
Output:
=======
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
{'query': 'Provide NVIDIAs outlook for the third quarter of fiscal 2024', 
'result': " of NVIDIA ’s underlying operating and technical performance.\n\nFor 
the period ended December 31, 2013, the Company is required to publish a Non-GAAP
measure of certain of its proprietary proprietary software packages.
........... WE HAVE TRUNCATED THE RESULT .......................................
New revenue increased by 3.1% and 3.2% for the three period ended December 31, 2014.
\n\nand. The non-GAAP non-GAAP non-GAAP measures also include non-inalliance
capital expenditure for the six months ended December 31, 2013, the twelve-month
fixed-cost-based accounting period beginning in the third quarter and to be
concluded in the fourth quarter, but the non-GAAP non-GAAP non-GAAP non-GAAP
measures do not include such capital expenditures. The non-GA"}
"""
