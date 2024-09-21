import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords

# Sample documents
documents = [
    "Natural language processing is a fascinating field in AI.",
    "Topic modeling helps uncover hidden themes in text data.",
    "Latent Dirichlet Allocation (LDA) is a popular topic modeling technique.",
    "LDA assumes that documents are mixtures of topics.",
    "Text mining and NLP are essential for extracting insights from text.",
    "Machine learning plays a significant role in NLP tasks.",
]

# Preprocess the documents (tokenization and lowercasing)
documents = [remove_stopwords(k) for k in documents]
documents = [doc.lower().split() for doc in documents]

# Create a dictionary and a document-term matrix (DTM)
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Build the LDA model
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# Print the topics
for topic in lda_model.print_topics():
    print(topic)

# To summarize the input
"""
(0, '0.062*"nlp" + 0.062*"text" + 0.037*"insights" + 0.037*"mining" + 0.037*"extracting" + 0.037*"essential" + 0.037*"text." + 0.037*"helps" + 0.037*"data." + 0.037*"themes"')
(1, '0.040*"modeling" + 0.040*"topic" + 0.040*"popular" + 0.040*"technique." + 0.040*"(lda)" + 0.040*"allocation" + 0.040*"dirichlet" + 0.040*"latent" + 0.040*"field" + 0.040*"natural"')

Here we have got 2 topics. 0 and 1. Both contains the words which are associated with the theme of the doc.
The words are arranged in their order. From left being most associated to right being least associated.
Based on the words we can say that Topic 0 is about natural language processing.
Topic 1 is about LDA method.
"""