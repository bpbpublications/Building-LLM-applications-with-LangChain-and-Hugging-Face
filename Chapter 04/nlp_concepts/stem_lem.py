import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import Word
from pattern.en import lemma

# ======================================================================
# NLTK
print("*" * 25)
print("Below example of Stemming using NLTK package")
nltk.download("punkt")  # Download necessary data (if not already downloaded)

# Create a PorterStemmer instance
stemmer = PorterStemmer()

# Example words for stemming
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform stemming on each word
stemmed_words = [stemmer.stem(word) for word in words]

# Print the original and stemmed words
for i in range(len(words)):
    print(f"Original: {words[i]}\tStemmed: {stemmed_words[i]}")


# ======================================================================
# NLTK
print("*" * 25)
print("Below example of Lemmatization using NLTK package")
nltk.download("wordnet")  # Download necessary data (if not already downloaded)

lemmatizer = WordNetLemmatizer()

# Example words for lemmatization
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform lemmatization on each word
lemma_words = [
    lemmatizer.lemmatize(word, pos="v") for word in words
]  # Specify the part of speech (e.g., 'v' for verb)

# Print the original and lemmatized words
for i in range(len(words)):
    print(f"Original: {words[i]}\tLemmatized: {lemma_words[i]}")


# ======================================================================
# SpaCy
print("*" * 25)
print("Below example of Lemmatization using Spacy package")

nlp = spacy.load("en_core_web_sm")

# Example words for lemmatization
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform lemmatization on each word
lemma_words = [nlp(word)[0].lemma_ for word in words]

# Print the original and lemmatized words
for i in range(len(words)):
    print(f"Original: {words[i]}\tLemmatized: {lemma_words[i]}")


# ======================================================================
# TextBlob
print("*" * 25)
print("Below example of Lemmatization using Textblob package")

# Example words for lemmatization
words = ["jumps", "jumping", "jumper", "flies", "flying"]

# Perform lemmatization on each word
lemma_words = [
    Word(word).lemmatize("v") for word in words
]  # Specify the part of speech (e.g., 'v' for verb)

# Print the original and lemmatized words
for i in range(len(words)):
    print(f"Original: {words[i]}\tLemmatized: {lemma_words[i]}")


# ======================================================================
# Pattern
# Not in use any more, since 2018 the package has not been updated.
# print("*" * 25)
# print("Below example of Lemmatization using Pattern package")

# # Example words for lemmatization
# words = ["jumps", "jumping", "jumper", "flies", "flying"]

# # Perform lemmatization on each word
# lemma_words = [lemma(word) for word in words]

# # Print the original and lemmatized words
# for i in range(len(words)):
#     print(f"Original: {words[i]}\tLemmatized: {lemma_words[i]}")
