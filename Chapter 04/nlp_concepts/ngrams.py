# Import required packages
from nltk.util import ngrams
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer


# ======================================================================
# NLTK
print("*" * 25)
print("Below example of N Grams is using NLTK package")
text = "This is an example sentence for creating n-grams."
n = 2  # Specify the n-gram size
bigrams = list(ngrams(text.split(), n))
print(bigrams)


# ======================================================================
# Spacy
print("*" * 25)
print("Below example of N Grams is using Spacy package")
# It is to download english package. Not required to run every time.
# Run below code from terminal after activating virtual environment
# ! python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
text = "This is an example sentence for creating n-grams."
n = 2  # Specify the n-gram size
tokens = [token.text for token in nlp(text)]
ngrams = [tokens[i : i + n] for i in range(len(tokens) - n + 1)]
print(ngrams)


# ======================================================================
# TextBlob
print("*" * 25)
print("Below example of N Grams is using TextBlob package")
# This is to download required corpora. Not required to run every time.
# Run below code from terminal after activating virtual environment
# python -m textblob.download_corpora
text = "This is an example sentence for creating n-grams."
n = 2  # Specify the n-gram size
blob = TextBlob(text)
bigrams = blob.ngrams(n)
print(bigrams)


# ======================================================================
# Scikit Learn
print("*" * 25)
print("Below example of N Grams is using Scikit Learn package")
# For scikit learn list is required hence providing list.
text = ["This is an example sentence for creating n-grams."]
n = 2  # Specify the n-gram size
vectorizer = CountVectorizer(ngram_range=(n, n))
X = vectorizer.fit_transform(text)
# Get the n-gram feature names
feature_names = vectorizer.get_feature_names_out()
# Print the n-grams
for feature_name in feature_names:
    print(feature_name)


# ======================================================================
# Hugging Face Package
print("*" * 25)
print("Below example of N Grams is using Hugging Face package")

# Define your text
text = "This is an example sentence for creating ngrams with Hugging Face Transformers."

# Choose a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text
tokens = tokenizer.tokenize(text)

# Generate bigrams
bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

# Generate trigrams
trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]

# Print the bigrams
for bigram in bigrams:
    print(bigram)

# Print the trigrams
for trigram in trigrams:
    print(trigram)
