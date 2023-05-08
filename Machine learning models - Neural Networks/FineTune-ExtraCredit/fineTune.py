## Fine Tune using pre-trained Word2Vec embeddings - Homework #3 - MoniKa Vyas CS 59000 14 NLP
######### PLEASE DOWNLOAD PRE-TRAINED WORD2VEC MODEL FROM : https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
import os
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.models import Word2Vec, KeyedVectors

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#nltk.download("stopwords") #Only need to run this once
#nltk.download('punkt')

# Get the current working directory path
current_dir_path = os.getcwd()

twitter_dataset = current_dir_path+r'\olid-training-v1.0.tsv'

# Preprocess the data.
def preprocessed_data(input_string):
    # Remove commonly used words i.e., stopwords. 
    stop_words = set(stopwords.words("english"))
    stop_words.add('vs.')
    stop_words.add('url')
    stop_words.add('re')
    stop_words.add('user')
    new_string = strip_punctuation(input_string)
    preprocessed_data = [x.strip(" '") for x in re.split(r"[^a-z'.]",new_string.lower())if len(x) > 0 and x not in stop_words]
    preprocess_data_string = " ".join(preprocessed_data)
    preprocessed_data = word_tokenize(preprocess_data_string)
    return preprocessed_data

# This method handles the evaluation of trained model.
# It returns accuracy, and classification report.
def evaluate_model(model,x,y):
  y_pred = model.predict(x)
  score_ = model.score(x,y)
  report = classification_report(y,y_pred)
  return score_,report

# `update_embeddings` method to fine tunes a pre-trained Word2Vec embeddings
# on the training text and save the fine-tuned embeddings.
# Returns the path to the fine-tuned word embeddings.
#-------------------------------------------------------------------------------------------------
# @parameters - 
#   `path_to_train_file`: path of file which can be used for
#                         training the pre-trained word2vec model,
#                         input file could be olid(train_file)
#-------------------------------------------------------------------------------------------------
def update_embeddings(path_to_train_file):
    # Read file one
    train_file = pd.read_csv(path_to_train_file, delimiter = '\t' )

    # Preprocessing file one.
    train_file['tweet'] = train_file['tweet'].apply(lambda x: preprocessed_data(x))
    
    # Extract pre-processed tweet values to train MLP model.
    preprocessed_tweets = train_file['tweet'].values

    # Load pre-trained Word2Vec model.
    # Please download the pre-trained word2vec model from here and
    # unzip it in order to run the program:
    # https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
    original_pretrained_model = KeyedVectors.load_word2vec_format(current_dir_path+r"\GoogleNews-vectors-negative300.bin", binary=True, limit=300000)

    # Pre-trained fine tuned model.
    finetuned_model = Word2Vec(vector_size=300, min_count=1)
    finetuned_model.build_vocab(preprocessed_tweets)

    # Save the vocab of the dataset
    vocab = list(finetuned_model.wv.key_to_index.keys())

    total_examples = finetuned_model.corpus_count
    finetuned_model.build_vocab([list(original_pretrained_model.index_to_key)], update=True)

    finetuned_model.wv.vectors_lockf = np.ones(len(finetuned_model.wv), dtype=np.float32)
    
    # Train the fine-tuned model over training dataset.
    finetuned_model.train(preprocessed_tweets, total_examples=total_examples, epochs=finetuned_model.epochs)
    
    # Save the fine-tuned embeddings.
    finetuned_model.save("fine_tuned_embeddings.txt")

    # Note:: The below code is to train the MLP model over fine-tuned embeddings.
    # It is in hidden mode just to avoid unnecessary confusion.
    # # word_embeddings = np.array([ finetuned_model.wv[k] if k in finetuned_model.wv else np.zeros(100) for k in vocab ])
    # # x = word_embeddings[0:13240, :]
    # # y = train_file['subtask_a'].values
    # # mlp_model = MLPClassifier(max_iter=2000)
    # # mlp_model.fit(x, y)
    # # return mlp_model

    # Return the path of the fine-tuned embeddings.
    return current_dir_path+r"\fine_tuned_embeddings.txt"

# This method is created to test the trained MLP model using fine tuning embeddings.
def test_MLP_model(path_to_test_file, MLP_model):
    #get the data file and create a pandas dataframe
    test_file_data = pd.read_csv(path_to_test_file, delimiter = '\t' )

    #clean the data using the function clean_data()
    test_file_data['tweet'] = test_file_data['tweet'].apply(lambda x: preprocessed_data(x))

    #get the full labels dataset
    labels_a = current_dir_path+r'\labels-levela.csv'
    labels_df_full = pd.read_csv(labels_a,names = ['id','subtask_a'],header = None)

    #Extract the test data
    preprocessed_tweets = test_file_data['tweet'].values

    original_pretrained_model = KeyedVectors.load_word2vec_format(current_dir_path+r"\GoogleNews-vectors-negative300.bin", binary=True, limit=300000)

    finetuned_model = Word2Vec(vector_size=300, min_count=1)
    
    finetuned_model.build_vocab(preprocessed_tweets)

    # Save the vocab of the dataset
    vocab = list(finetuned_model.wv.key_to_index.keys())

    total_examples = finetuned_model.corpus_count
    finetuned_model.build_vocab([list(original_pretrained_model.index_to_key)], update=True)

    finetuned_model.wv.vectors_lockf = np.ones(len(finetuned_model.wv), dtype=np.float32)

    finetuned_model.train(preprocessed_tweets, total_examples=total_examples, epochs=finetuned_model.epochs)

    word_embeddings = np.array([ finetuned_model.wv[k] if k in finetuned_model.wv else np.zeros(100) for k in vocab ])
    X_test = word_embeddings[0:860, :]
    # extract the classifications
    y_test = labels_df_full['subtask_a'].values
    # get the data from the classifier
    score, report = evaluate_model(MLP_model,X_test,y_test) 

    print(report)
    print('score:',score)