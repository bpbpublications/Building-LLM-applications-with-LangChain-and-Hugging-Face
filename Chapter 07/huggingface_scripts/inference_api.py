"""
This script will provide an overview that how to work with huggingface API
https://huggingface.co/docs/api-inference/quicktour

First you need to define the model to be used from https://huggingface.co/models
and at last put that model id at the end of the BASE_API_URL

You can get list of parameters that you can utilize with APIs for respective tasks on below URL.
https://huggingface.co/docs/api-inference/detailed_parameters
"""

import requests

# Common parameters
API_TOKEN = "PUT_HUGGINGFACE_TOKEN_HERE"
BASE_API_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

Q1 = "Explain Large Language Models in funny way so that child can understand."
Q2 = "What is cricket provide brief details."


def query(API_URL: str, headers: dict, payload: str) -> dict:
    """
    Function to get response from API

    :param API_URL: str
        URL of the API to get the response
    :param headers: dict
        Headers to be used in API call
    :param payload: str
        Paylod which will contain query
    :return: dict
    """
    payload = {"inputs": payload}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# ======================================================================================================================
# Text Generation Models & Usage
# ======================================================================================================================

# ......................................................................................................................
# GPT2 model
gpt2_url = "https://api-inference.huggingface.co/models/gpt2"
q2_gpt2_ans = query(API_URL=gpt2_url, headers=headers, payload=Q2)
print(q2_gpt2_ans)

"""
Output:
-------
[{'generated_text': "What is cricket provide brief details. From the theme characteristics that
helped define the focus of cricket to how to identify topics and groups, the interesting bits are not
much further than the understories of the game. One of the goals of many weavers, although not exclusively
focussing on the superficial, is to bring point to sequence without sounding grandiose. Studies find
the three influential aspects to cricket use are in look at issues and factors, and in understanding
the technology, so that people don't succumb to them. An analogy to"}]
"""

# ......................................................................................................................
# Dolly model
dolly_url = "https://api-inference.huggingface.co/models/databricks/dolly-v2-3b"
q2_dolly_ans = query(API_URL=dolly_url, headers=headers, payload=Q2)
print(q2_dolly_ans)

"""
Output:
-------
[{'generated_text': 'What is cricket provide brief details.\nCricket: Australian
Rules Football. The game involves two teams of contesting players who running around
a 70-metre curved oval with a slightly rotated baseball diamond. The objective is to
get the ball into the southern end of the oval, where a designated goal may be
supported by two posts, called wickets. The team successful in getting the ball into
the oval from the opposing end are the winners. The scoring mechanism is similar to
any footy match, with the ball carrying a small'}]
"""
