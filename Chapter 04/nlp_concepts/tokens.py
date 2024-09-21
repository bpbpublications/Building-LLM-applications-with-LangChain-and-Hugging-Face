# Import required packages
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from transformers import AutoTokenizer
from textblob import TextBlob


# ======================================================================
# NLTK
print("*" * 25)
print("Below example of Tokens is using NLTK package")

# Download the required dataset. Not required to run everytime.
nltk.download("punkt")
text = "This is an example sentence. Tokenize it."

# Word tokenization
words = word_tokenize(text)
print("Word tokens:", words)

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentence tokens:", sentences)


# ======================================================================
# Spacy
print("*" * 25)
print("Below example of Tokens is using Spacy package")

# It is to download english package. Not required to run every time.
# Run below code from terminal after activating virtual environment
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

text = "This is an example sentence. Tokenize it."

doc = nlp(text)

# Word tokenization
words = [token.text for token in doc]
print("Word tokens:", words)

# Sentence tokenization
sentences = [sent.text for sent in doc.sents]
print("Sentence tokens:", sentences)


# ======================================================================
# Builtin Methods
print("*" * 25)
print("Below example of Tokens is using Builtin package")

text = "This is an example sentence. Tokenize it."

# Word tokenization
words = text.split(" ")
print("Word tokens:", words)

# Sentence tokenization
sentences = text.split(".")
# Remove 3rd element which will be "". Also remove extra spaces around non-blank elements.
sentences = [k.strip() for k in sentences if k != ""]
print("Sentence tokens:", sentences)


# ======================================================================
# Huggingface Transformers
print("*" * 25)
print("Below example of Tokens is using Huggingface package")

# Use pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "This is an example sentence. Tokenize it."

# Tokenize the text into word-level tokens
word_tokens = tokenizer.tokenize(text)
print("Word tokens:", word_tokens)

# we tokenize the text into sentence-level tokens by adding special tokens (e.g., [CLS] and [SEP]) to the output.
# [CLS] stands for Classification Token and used in BERT and other transformers for classification tasks. Its also
# inserted at the beginning of text sequence.
# [SEP] stands for Separator Token and used in BERT and other transformers. It is used to separate different segments
# of the input text.
# Tokenize the text into sentence-level tokens
sent_tokens = tokenizer.tokenize(text, add_special_tokens=True)
print("Sentence tokens:", sent_tokens)

# Optionally, you can convert the sentence tokens into actual sentences
sentences = tokenizer.convert_tokens_to_string(sent_tokens)
print("Sentences:", sentences)


# ======================================================================
# Textblob
print("*" * 25)
print("Below example of Tokens is using Textblob package")

text = "This is an example sentence. Tokenize it."

blob = TextBlob(text)

# Word tokenization
words = blob.words
print("Word tokens:", words)

# Sentence tokenization
sentences = blob.sentences
print("Sentence tokens:", sentences)
