"""
This script will demonstrate how to use Python transformer package for text generation.
https://huggingface.co/docs/transformers/pipeline_tutorial
https://huggingface.co/docs/transformers/llm_tutorial
https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.TextGenerationPipeline

Get list of models from https://huggingface.co/models

Please note that for publicly available models the token is not required.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

token = "PUT_HUGGINGFACEHUB_TOKEN_HERE"

Q1 = "Explain Large Language Models in funny way so that child can understand."
Q2 = "What is cricket provide brief details."

# If the parameter size is big i.e. > 7B need to provide this argument offload_folder="offload"
# Else it will raise an error. Here its for representation purpose only.
# ValueError: The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder` for them.
# Alternatively, make sure you have `safetensors` installed if the model you are using offers the weights in this format

# This is the First way to use LLM by transformer package...............................................................
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

print(dolly_generate_text(Q1))

"""
Output:
------
[{'generated_text': 'Explain Large Language Models in funny way so that child can
understand.\nLarge Language Models are computers programs that are capable of
understanding human languages. In order to understand human languages, one needs to
have a lot of data. Languages are very similar but not identical. Words can have the
same meaning but mean a completely different thing in each language. This is why
learning multiple languages is so difficult for humans. To teach computers how to
understand languages, we use Languages called Natural Language Processing. These
programs typically follow steps to process the human language. First, they split
the human language into smaller parts called words. These words are very similar,
therefore the program needs to find words using pattern recognition. Words are then
joined back together to form sentences. A sentence does not need to have to make sense,
it just has to be a combination of words. Finally, the program notifies the human if
there is an error in the sentence. This way, a computer program will be able to
understand human languages.'}]
"""


# This is the Second way to use LLM by transformer package .............................................................
# With Auto classes like AutoTokenizer, AutoModelForCausalLM we will get more low level access.
# With Pipeline, we will have high level access. Again pipeline uses Auto Classes.
model_id = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir="E:\\Repository\\Book\\models",
    token=token,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="E:\\Repository\\Book\\models",
    device_map="auto",
    offload_folder="offload",
    token=token,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
)
print(pipe(Q2))

"""
Output:
------
[{'generated_text': 'What is cricket provide brief details.\nCricket is a game played between two teams of eleven 
players each. The game is played on a rectangular pitch of size 100 yards (100 meters) by 40 yards (30 meters). The game
is played with a bat and a ball. The bat has three main parts - a handle, a barrel and a blade. The ball has two main 
parts - a leather ball and a coating of rubber on the ball. The game is played with a number of players from both 
sides. The players'}]
"""

# Generate text using the model.........................................................................................
# this way as well we can generate the text
# it gives us more minute control in setting the parameters at low level similar to above second method.

inputs = tokenizer(Q2, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)

# Decode and print the output
text = tokenizer.batch_decode(outputs)[0]
print(text)
"""
Output:
------
What is cricket provide brief details.
Cricket is a game played between two teams of eleven players each. The game is played on a rectangular pitch of size 100
yards (100 meters) by 40 yards (30 meters). The game is played with a bat and a ball. The bat has three main parts - a
handle, a barrel and a blade. The ball has two main parts - a leather ball and a coating of rubber on the ball. The game
is played with a number of players from both sides. The players take turns to bat and bowl. The batsman can hit the ball
only when the ball is moving. The bowler can bowl the ball only when the bat is not moving. The game is played with a
number of rules. The game is played with a number of rules. The game is played with a number of rules. The game is
played with a number of rules. The game is played with a number of rules. The game is played with a number
"""
