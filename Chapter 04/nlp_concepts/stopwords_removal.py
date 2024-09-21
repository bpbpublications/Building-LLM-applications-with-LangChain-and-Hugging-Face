import nltk
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# In case you get an error "ImportError: cannot import name 'triu' from 'scipy.linalg'"
# when importing Gensim, please install specific version of scipy
# pip install scipy==1.12

# ======================================================================
# NLTK
print("*" * 25)
print("Below example of Stop Words Removal using NLTK package")

nltk.download("stopwords")  # Download necessary data (if not already downloaded)

text = "This is an example sentence with some stop words."

words = text.split()
filtered_words = [
    word for word in words if word.lower() not in stopwords.words("english")
]

print("Without Stop Words!!!")
print(filtered_words)


# ======================================================================
# Spacy
print("*" * 25)
print("Below example of Stop Words Removal using Spacy package")

nlp = spacy.load("en_core_web_sm")

text = "This is an example sentence with some stop words."

doc = nlp(text)

filtered_words = [token.text for token in doc if not token.is_stop]

print("Without Stop Words!!!")
print(filtered_words)


# ======================================================================
# Gensim
print("*" * 25)
print("Below example of Stop Words Removal using Gensim package")

text = "This is an example sentence with some stop words."

filtered_text = remove_stopwords(text)

print("Without Stop Words!!!")
print(filtered_text)


# ======================================================================
# Scikit Learn
print("*" * 25)
print("Below example of Stop Words Removal using Scikit-Learn package")

text = "This is an example sentence with some stop words."

words = text.split()
filtered_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]

print("Without Stop Words!!!")
print(filtered_words)
