from textblob import TextBlob
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ======================================================================
# TextBlob
print("*" * 25)
print("Below example of Sentiment using TextBlob package")

# Sample text for sentiment analysis
text = "I love this product! It's amazing."

# Create a TextBlob object
blob = TextBlob(text)

# Perform sentiment analysis
sentiment = blob.sentiment

# Print sentiment polarity and subjectivity
polarity = sentiment.polarity  # Range from -1 (negative) to 1 (positive)
subjectivity = sentiment.subjectivity  # Range from 0 (objective) to 1 (subjective)

# Interpret sentiment
if polarity > 0:
    sentiment_label = "positive"
elif polarity < 0:
    sentiment_label = "negative"
else:
    sentiment_label = "neutral"

# Output results
print("Text:", text)
print("Sentiment Polarity:", polarity)
print("Sentiment Subjectivity:", subjectivity)
print("Sentiment Label:", sentiment_label)


# ======================================================================
# HuggingFace
print("*" * 25)
print("Below example of Sentiment using HuggingFace package")

# Load a pre-trained sentiment analysis model
nlp = pipeline("sentiment-analysis")

# Sample text for sentiment analysis
text = "I love this product! It's amazing."

# Perform sentiment analysis
results = nlp(text)

# Output results
for result in results:
    label = result["label"]
    score = result["score"]
    print(f"Sentiment Label: {label}, Score: {score:.4f}")


# ======================================================================
# NLTK
print("*" * 25)
print("Below example of Sentiment using NLTK package")

# Download the VADER lexicon (if not already downloaded)
nltk.download("vader_lexicon")

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample text for sentiment analysis
text = "I love this product! It's amazing."

# Perform sentiment analysis
sentiment = analyzer.polarity_scores(text)

# Interpret sentiment
compound_score = sentiment["compound"]
if compound_score >= 0.05:
    sentiment_label = "positive"
elif compound_score <= -0.05:
    sentiment_label = "negative"
else:
    sentiment_label = "neutral"

# Output results
print("Text:", text)
print("Sentiment Score:", sentiment)
print("Sentiment Label:", sentiment_label)
