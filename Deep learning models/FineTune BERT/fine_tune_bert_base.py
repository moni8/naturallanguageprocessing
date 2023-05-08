## Fine Tuned BERT Base uncased Model - Homework #4 - MoniKa Vyas CS 59000 14 NLP

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import nlp

# # BERT base uncased model.
model_name = "bert-base-uncased"

# # Get the current working directory path
current_dir_path = os.getcwd()

# Read OLID training dataset file.
path_to_train_file = "olid-training-v1.0.tsv"
train_dataset = pd.read_csv(path_to_train_file, delimiter = '\t')

# convert pandas dataframe into a pyarrow object.
tr_ds = nlp.Dataset.from_pandas(train_dataset)

# Remove unnecessary column from OLID training dataframe,and rename
# `subtask_a` to `label` to be able to process by the BERT base uncased model.
tr_ds.remove_column_("id")
tr_ds.remove_column_("subtask_b")
tr_ds.remove_column_("subtask_c")
tr_ds.rename_column_("subtask_a","label")

# convert `string` labels to `int`.
label_to_int = {
    "OFF": 0,
    "NOT": 1
}
tr_ds = tr_ds.map(lambda example: {"label": label_to_int[example["label"]]})

# Define tokenzier for the pre-trained model.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convert OLID training dataset into the pre-trained transformer model
# readable format that could be processed by the mdoel
# using `tokenzier` loaded from `AutoTokenizer` method.
encoded_dataset = tr_ds.map(lambda examples: tokenizer(examples['tweet'], padding="max_length", truncation=True, max_length=512), batched=True)
encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Smaller subset of the full dataset to fine-tune on to reduce the time it takes.
small_train_dataset = encoded_dataset.shuffle(seed=42).select(range(6000))

# Pre-process test dataset.
path_to_test_file = 'testset-levela.tsv'
test_dataset = pd.read_csv(path_to_test_file)
test_ds = nlp.Dataset.from_pandas(test_dataset)
test_ds.remove_column_("id")
test_ds = test_ds.map(lambda example: {"label": label_to_int[example["label"]]})
encoded_test_dataset = test_ds.map(lambda examples: tokenizer(examples['tweet'], padding="max_length", truncation=True, max_length=512), batched=True)

# Define pre-trained transformer model BERT Base uncased.
# loaded using `from_pretained` method.
config = AutoConfig.from_pretrained(model_name, num_lables=2)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

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

# Define Trainer arguments, `output directory` and `number of epochs`.
args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1
)

# `Trainer` a Pytorch class API for feature-complete training is
# used for the pre-trained tansformer model. 
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train_dataset,
    compute_metrics=compute_metrics
)

# fine-tune your model by calling train()
trainer.train()

# evaluate fine-tuned model.
eval_result = trainer.evaluate(eval_dataset=encoded_test_dataset)

# Display evaluation results.
for key,item in eval_result.items():
    print(key," : ",item)

# Make prediction
raw_pred, _, _ = trainer.predict(encoded_test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

# helper function to format and output the
# fine-tuned BERT predicitons to the file `Bert_base_model_results.tsv`
def format_and_output_test_File(test_dataframe,prediction):
  df = test_dataframe
        
  df['prediction'] = prediction

  #output the file
  df.to_csv('Bert_base_model_results'+'.tsv',sep="\t",index=False)

# Create the output test file with prediction labels.
format_and_output_test_File(test_dataset,y_pred)
