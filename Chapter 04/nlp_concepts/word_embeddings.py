from gensim.models import Word2Vec
import spacy
from transformers import DistilBertTokenizer, DistilBertModel


# ======================================================================
# Gensim
print("*"*25)
print("Below example of Word Embeddings using Gensim package")

# Example sentences for training the model
sentences = [
    "This is an example sentence for word embeddings.",
    "Word embeddings capture semantic relationships.",
    "Gensim is a popular library for word embeddings.",
]

# Tokenize the sentences
tokenized_sentences = [sentence.split() for sentence in sentences]

# Train a Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# Access word vectors
word_vector = model.wv['word']
print(word_vector)


# ======================================================================
# Spacy
print("*"*25)
print("Below example of Word Embeddings using Spacy package")

# Load the pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Process a text to get word embeddings
doc = nlp("This is an example sentence for word embeddings. Word embeddings capture semantic relationships. Gensim is a popular library for word embeddings.")
word_vector = doc[0].vector  # Access the word vector
print(word_vector)


# ======================================================================
# Huggingface
print("*"*25)
print("Below example of Word Embeddings using Huggingface package")

# Load the pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize a sentence
text = "Hugging Face's Transformers library is fantastic!"
tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Load the pre-trained DistilBERT model
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Get word embeddings for the tokens
output = model(**tokens)

# Access word embeddings for the [CLS] token (you can access other tokens as well)
word_embeddings = output.last_hidden_state[0]  # [CLS] token's embeddings

# Convert the tensor to a numpy array
word_embeddings = word_embeddings.detach().numpy()

# Print the word embeddings
print(word_embeddings)
