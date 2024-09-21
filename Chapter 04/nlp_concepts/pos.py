# Import required packages
import nltk
import spacy
from textblob import TextBlob


# ======================================================================
# NLTK
print("*"*25)
print("Below example of POS using NLTK package")

nltk.download('punkt')  # Download necessary data (if not already downloaded)

text = "This is an example sentence for part-of-speech tagging."
words = nltk.word_tokenize(text)
tagged_words = nltk.pos_tag(words)

print(tagged_words)


# ======================================================================
# Spacy
print("*"*25)
print("Below example of POS using Spacy package")

nlp = spacy.load("en_core_web_sm")

text = "This is an example sentence for part-of-speech tagging."
doc = nlp(text)

for token in doc:
    print(token.text, token.pos_)


# ======================================================================
# TextBlob
print("*"*25)
print("Below example of POS using TextBlob package")

text = "This is an example sentence for part-of-speech tagging."
blob = TextBlob(text)

for word, pos in blob.tags:
    print(word, pos)
