"""
To get an overview of how transfer learning works
"""

# Importing the necessary module from the transformers library
from transformers import pipeline

# Creating a zero-shot classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    device_map="auto",  # Automatically distributes the model across available GPUs and CPUs
    model_kwargs={
        "cache_dir": "E:\\Repository\\Book\\models",
        "offload_folder": "offload",  # use it when model size is > 7B
    },
)

# Input text for classification
text = "This article discusses transfer learning for zero-shot text categorization."

# Candidate labels that the model will consider
candidate_labels = ["machine learning", "natural language processing", "data science"]

# Performing zero-shot classification on the input text with the candidate labels
results = classifier(text, candidate_labels)

print(results)
"""
Output:
-------
{'sequence': 'This article discusses transfer learning for zero-shot text categorization.',
 'labels': ['machine learning', 'natural language processing', 'data science'],
 'scores': [0.46083739399909973, 0.3666556179523468, 0.17250701785087585]}
"""

# Displaying the results individually
for rng in range(len(results["labels"])):
    # Printing the predicted label and its associated confidence score
    print(f"Label: {results['labels'][rng]}")
    print(f"Score: {results['scores'][rng]:.4f}")

"""
Output
Label: machine learning
Score: 0.4608
Label: natural language processing
Score: 0.3667
Label: data science
Score: 0.1725
"""
