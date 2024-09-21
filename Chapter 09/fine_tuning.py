# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:05:15 2023
"""
## Importing necessary libraries ####

import os
import torch
import numpy as np
import pandas as pd
from time import time
from datasets import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# Change directory where we have placed the data.
os.chdir(r"C:\\projects\\actual\\2023\\bedrock\\data\\fine_tuning")

## Reading the jsonl file for training ####
df = pd.read_json("sport2.jsonl", lines=True)  # we are going to use the same file
df.head()

# replacing line space and new line
df = df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["", ""], regex=True)
df.head()
df.columns

### using Cuda device
device = "cuda" if torch.cuda.is_available() else "cpu"

##Changing column names as model expects data column as text and target variable as labels ##
##df.columns = ['text','label']

# train the label encoder , convert the categories to numeric features
le = preprocessing.LabelEncoder()
le.fit(df["news_category"])

le.classes_
len(le.classes_)
df["label"] = le.transform(df["news_category"])
df["label"].unique()
df.reset_index(inplace=True)

# Saving the label encoder to a numpy file for reusability
PATH = r"path of your folder where you want to save the data"
np.save(PATH + "label_encoder_news_category.npy", le.classes_)

##  re load the encoder
PATH = r"path of your folder where data is saved"
le = preprocessing.LabelEncoder()
le.classes_ = np.load(PATH + "label_encoder_domain_whole.npy", allow_pickle=True)

# Get training and testing data splitted
train_df, test_df = train_test_split(
    df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"]
)
train_df.head()


## we are going to use open source BERT Base model from hugginface
# default storage directory will be ~/.cache/
# you can add/update other hyper parameters as well as per the requirement.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(le.classes_)
)  # change the number of labels
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

## converting dataset to huggingface dataset
train_df_ar = Dataset.from_pandas(train_df)
test_df_ar = Dataset.from_pandas(test_df)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


train_df_tf = train_df_ar.map(tokenize, batched=True, batch_size=len(train_df_ar))
test_df_tf = test_df_ar.map(tokenize, batched=True, batch_size=len(test_df_ar))
train_df_tf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_df_tf.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# to free up GPU memory
torch.cuda.empty_cache()


# starting the training process
# training parameters
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",
    save_total_limit=1,
    # load_best_model_at_end=True
    # directory for storing logs
)


##the instantiated 🤗 Transformers model to be trained
trainer = Trainer(
    model=model,  # the instantiated  model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_df_tf,  # training dataset
    eval_dataset=test_df_tf,  # evaluation dataset
)

start = time()

# It will start the training process
trainer.train()

end = time()

total = end - start
print(f"time taken by the process is {total/60} minutes ")

## this will run all the evaluation metrics and provide the results
trainer.evaluate()
"""
Output:
=====
{'eval_loss': 0.23266847431659698,
'eval_runtime': 51.081,
'eval_samples_per_second': 64.27,
'eval_steps_per_second': 2.016,
'epoch': 3.0}
    
{'eval_loss': 0.20599809288978577,
'eval_runtime': 53.0823,
'eval_samples_per_second': 61.847,
'eval_steps_per_second': 1.94,
'epoch': 3.0}
"""

# Saving the model to a folder domain_classification
trainer.save_model("./results/domain_classification")
