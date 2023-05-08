import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Define pre-trained transformer model for text classification task.
# This is a Roberta base version model loaded using `from_pretained` method.
model_name = "aychang/roberta-base-imdb"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define tokenzier for the pre-trained model.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Method to evalaute certain evaluation metrices.
# It returns accuracy, recall, precision, and F1-score.
#-----------------------------------------------------------------------------------
# @parameters: 
#       prediction: It holds the predicted labels and the test labels,
#                   which can be used to compute the metrices for the given model.
#-----------------------------------------------------------------------------------
def compute_metrics(pred):

    pred, labels = pred
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "F1":f1}

# `Trainer` a Pytorch class API for feature-complete training is
# used for the pre-trained tansformer model. 
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

# Load test data using `load_dataset` method to evalaute
# the pre-trained transformer model.
# The test dataset that is used here is also from IMDB, but with different entries
# IMDB test dataset loaded from here: https://huggingface.co/datasets/noob123/imdb_test
emotion_train_dataset = load_dataset("noob123/imdb_test", split="test")

# Select only 100 samples of test dataset to reduce
# the time of testing the pre-trained transformer model.
emotion_train_dataset = emotion_train_dataset.select(range(100))

# Convert test dataset into the pre-trained transformer model
# readable format that could be processed by the mdoel
# using `tokenzier` loaded from `AutoTokenzer` method.
encoded_dataset = emotion_train_dataset.map(lambda examples: tokenizer(examples['reviews'], padding=True, truncation=True, max_length=512), batched=True)

# `Trainer` class extended to evaluate the pre-trained model.
eval_result = trainer.evaluate(eval_dataset=encoded_dataset)

# Display pre-trained transformer model metrics.
for key,item in eval_result.items():
    if (key == "eval_loss") or (key == "eval_accuracy") or (key == "eval_F1"):
        print(key,": ",item)
