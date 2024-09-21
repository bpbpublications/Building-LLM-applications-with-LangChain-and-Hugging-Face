import nltk
import spacy
import subprocess
from transformers import pipeline

# ======================================================================
# NLTK
print("*" * 25)
print("Below example of NER is using NLTK package")

# This is one time only. Not required to run every time.
# Once you have got the chunker you can comment the code.
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("averaged_perceptron_tagger")

# Run below code from terminal after activating virtual environment
""" python -m spacy download en_core_web_sm """

text = "Apple Inc. is headquartered in Cupertino, California, and was founded by Steve Jobs."

words = nltk.word_tokenize(text)
tagged_words = nltk.pos_tag(words)
entities = nltk.chunk.ne_chunk(tagged_words)

for entity in entities:
    if isinstance(entity, nltk.Tree):
        print([(word, entity.label()) for word, tag in entity])


# ======================================================================
# Spacy
print("*" * 25)
print("Below example of NER is using Spacy package")

nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is headquartered in Cupertino, California, and was founded by Steve Jobs."

doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)


# ======================================================================
# HuggingFace
print("*" * 25)
print("Below example of NER is using HuggingFace package")

# Load the NER model
# It will download large model of size around 1.33 GB
"""
If you are getting error as mentioned below uninstall keras and tensorflow packages.
pip uninstall keras tensorflow

1. RuntimeError: Failed to import transformers.models.bert.modeling_tf_bert because of the following error (look up to see its traceback):
Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. Please install the backwards-compatible tf-keras package with `pip install tf-keras`.

2. RuntimeError: Failed to import transformers.models.bert.modeling_tf_bert because of the following error (look up to see its traceback):
module 'tensorflow._api.v2.compat.v2.__internal__' has no attribute 'register_load_context_function'
"""
nlp_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Apple Inc. is headquartered in Cupertino, California, and was founded by Steve Jobs."

# Perform NER
entities = nlp_ner(text)

# Display the detected entities
for entity in entities:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
