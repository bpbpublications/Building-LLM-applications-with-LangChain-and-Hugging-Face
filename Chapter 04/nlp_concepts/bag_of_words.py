import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from gensim.corpora import Dictionary
from collections import defaultdict
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ======================================================================
# Download required data
nltk.download("punkt")


# ======================================================================
# NLTK
print("*" * 25)
print("Below example of Bag Of Words is using NLTK package")
text = (
    "This is a sample document. Another document with some words. Repeating document with some words. A third "
    "document for illustration. Repeating illustration."
)
words = word_tokenize(text)
fdist = FreqDist(words)

fdist.pprint()


# ======================================================================
# Gensim
print("*" * 25)
print("Below example of Bag Of Words is using Gensim package")

documents = [
    "This is a sample document.",
    "Another document with some words. Repeating document with some words.",
    "A third document for illustration. Repeating illustration.",
]

tokenized_docs = [doc.split() for doc in documents]

# Create a dictionary
dictionary = Dictionary(tokenized_docs)

word_frequencies = dictionary.cfs

# Display words and their frequencies
for word_id, frequency in word_frequencies.items():
    word = dictionary[word_id]  # Get the word corresponding to the word ID
    print(f"ID: {word_id}, Word: {word}, Frequency: {frequency}")

# Create a BoW representation
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Create a TF-IDF model based on the BoW representation
tfidf = TfidfModel(corpus, dictionary=dictionary)

# Calculate overall TF-IDF scores for words
overall_tfidf = defaultdict(float)
for doc in tfidf[corpus]:
    for word_id, tfidf_score in doc:
        overall_tfidf[word_id] += tfidf_score

# Display words and their overall TF-IDF scores
for word_id, tfidf_score in overall_tfidf.items():
    word = dictionary[word_id]  # Get the word corresponding to the word ID
    print(f"Word: {word}, Overall TF-IDF Score: {tfidf_score:.4f}")


# ======================================================================
# Scikit Learn
print("*" * 25)
print("Below example of Bag Of Words is using Scikit-Learn package Count Method")

documents = [
    "This is a sample document.",
    "Another document with some words. Repeating document with some words.",
    "A third document for illustration. Repeating illustration.",
]

# Join the list of documents into a single string
corpus = " ".join(documents)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([corpus])

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Get the word frequencies from the CountVectorizer's array
word_frequencies = X.toarray()[0]

# Print words with their frequencies
for word, frequency in zip(feature_names, word_frequencies):
    print(f"Word: {word}, Frequency: {frequency}")


# ======================================================================
# Scikit Learn with TFIDF
print("*" * 25)
print("Below example of Bag Of Words is using Scikit-Learn package TFIDF Method")

documents = [
    "This is a sample document.",
    "Another document with some words. Repeating document with some words.",
    "A third document for illustration. Repeating illustration.",
]

# Join the list of documents into a single string
corpus = " ".join(documents)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform([corpus])

# Get the feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get the TF-IDF values from the TF-IDF vector
tfidf_values = X.toarray()[0]

# Print words with their TF-IDF values
for word, tfidf in zip(feature_names, tfidf_values):
    print(f"Word: {word}, TF-IDF: {tfidf:.4f}")
