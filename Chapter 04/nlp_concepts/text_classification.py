from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import SklearnClassifier
import random  # Import the random module

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# ======================================================================
# Scikit-Learn
print("*"*25)
print("Below example of Text Analysis using Sklearn package")
# Sample text data and labels
texts = ["This is a positive sentence.", "This is a negative sentence.", "A neutral statement here."]
labels = ["positive", "negative", "neutral"]

# Text preprocessing and feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a classifier (e.g., Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)


# ======================================================================
# NLTK
print("*"*25)
print("Below example of Text Analysis using NLTK package")

# Load the movie reviews dataset
# nltk.download('movie_reviews')
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Text preprocessing and feature extraction
all_words = [w.lower() for w in movie_reviews.words()]
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

feature_sets = [(find_features(rev), category) for (rev, category) in documents]

# Split data into training and testing sets
training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

# Train a classifier (e.g., Naive Bayes)
classifier = SklearnClassifier(MultinomialNB())
classifier.train(training_set)

# Evaluate the classifier
accuracy = nltk.classify.accuracy(classifier, testing_set)
print(f"Accuracy: {accuracy:.2f}")


# ======================================================================
# Hugging Face
print("*"*25)
print("Below example of Text Analysis using Hugging Face package")

# Sample text data
texts = ["This is a positive sentence.", "This is a negative sentence.", "A neutral statement here."]

# Preprocess text and load pre-trained model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the text
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Perform text classification
outputs = model(**inputs)

# Get predicted labels and probabilities
logits = outputs.logits
predicted_labels = torch.argmax(logits, dim=1)

# Map predicted labels to human-readable class names
class_names = ['positive', 'negative', 'neutral']

for i, text in enumerate(texts):
    print(f"Text: {text}")
    print(f"Predicted Label: {class_names[predicted_labels[i]]}")
    print("")

# You can also extract the probability scores for each class if needed
class_probabilities = torch.softmax(logits, dim=1)

