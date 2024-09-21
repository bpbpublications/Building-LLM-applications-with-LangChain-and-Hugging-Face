"""
The telegram bot related code is taken from https://github.com/cmd410/OrigamiBot
and then modified with our LLM bot to have conversation with users
"""

from sys import argv
from time import sleep
from origamibot import OrigamiBot as Bot
from origamibot.listener import Listener
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, pipeline
from langchain.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

MAX_MESSAGE_LENGTH = 4095  # Maximum length for a Telegram message


def split_message(message):
    """Split a message into chunks of maximum length."""
    return [
        message[i : i + MAX_MESSAGE_LENGTH]
        for i in range(0, len(message), MAX_MESSAGE_LENGTH)
    ]


# ======================================================================================================================
# Defining global settings for easy and fast work

# load text embedding model from HuggingFaceHub to generate vector embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    cache_folder="E:\\Repository\\Book\\sentence_transformers",
    model_kwargs={"device": "cpu"},  # make it to "cuda" in case of GPU
    encode_kwargs={"normalize_embeddings": False},
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
    token="PUT_HERE_HUGGINGFACEHUB_API_TOKEN",
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


# telegram related stuff -----------------------------------------------------------------------------------------------
class BotsCommands:
    """
    This are the commands which you can use in chat like..........
    /start will start the conversation
    /echo will echo the message
    """

    def __init__(self, bot: Bot):  # Can initialize however you like
        self.bot = bot

    def start(self, message):  # /start command
        self.bot.send_message(message.chat.id, "Hello user!\nThis is an example bot.")

    def echo(self, message, value: str):  # /echo [value: str] command
        self.bot.send_message(message.chat.id, value)

    def _not_a_command(self):  # This method not considered a command
        print("I am not a command")


class MessageListener(Listener):  # Event listener must inherit Listener
    """
    This is the message listener. Based on the question this portion will be
    answer. This will be responsible for conversation with user.
    """

    def __init__(self, bot):
        self.bot = bot
        self.m_count = 0

    def on_message(self, message):  # called on every message
        self.m_count += 1
        print(f"Total messages: {self.m_count}")
        ans = retrievalQA.invoke(message.text)
        chunks = split_message(ans["result"])
        for chunk in chunks:
            self.bot.send_message(message.chat.id, chunk)

    def on_command_failure(self, message, err=None):  # When command fails
        if err is None:
            self.bot.send_message(message.chat.id, "Command failed to bind arguments!")
        else:
            self.bot.send_message(message.chat.id, f"Error in command:\n{err}")


if __name__ == "__main__":
    token = argv[1] if len(argv) > 1 else input("Enter bot token: ")
    bot = Bot(token)  # Create instance of OrigamiBot class

    # Add an event listener
    bot.add_listener(MessageListener(bot))

    # Add a command holder
    bot.add_commands(BotsCommands(bot))

    # We can add as many command holders
    # and event listeners as we like

    bot.start()  # start bot's threads
    print("*" * 25)
    print("Bot has been started!!!")
    while True:
        sleep(1)
        # Can also do some useful work i main thread
        # Like autoposting to channels for example
