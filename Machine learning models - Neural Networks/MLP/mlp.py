## MLP - Homework #3 - MoniKa Vyas CS 59000 14 NLP

########## Note:: Acurracy is still not consisent, each time the model is trained,
# accuracy is different every time ############

import os
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.models import word2vec

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from sklearn.metrics import classification_report

#nltk.download("stopwords") #Only need to run this once
#nltk.download('punkt')

# Get the current working directory path
current_dir_path = os.getcwd()

twitter_dataset = current_dir_path+r'\olid-training-v1.0.tsv'

# Preprocess the data.
def preprocessed_data(input_string):    
    new_string = strip_punctuation(input_string)
    preprocessed_data = [x.strip(" '") for x in re.split(r"[^a-z'.]",new_string.lower())if len(x) > 0]
    preprocess_data_string = " ".join(preprocessed_data)
    preprocessed_data = word_tokenize(preprocess_data_string)
    return preprocessed_data

# Tried simple ways to pre-process training data for better accuracy.
def simple_preprocessing(raw_text, remove_stopwords=False):
    # 1. Remove non-letters, but including numbers
    letters_only = re.sub("[^0-9a-zA-Z]", " ", raw_text)
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # Remove stop words
        meaningful_words = [w for w in words if not w in stops]
        words = meaningful_words
    return words 

# `evaluate_model` method to handle probability, prediction of each test texts,
# calculate score and create report.
# It returns the prediction, score and report.
def evaluate_model(model,x,y):
  p_pred = model.predict_proba(x)
  y_pred = model.predict(x)
  score_ = model.score(x,y)
  report = classification_report(y,y_pred)
  return p_pred,y_pred,score_,report

# helper function to format and output the file for method test_LR_model
def format_and_output_test_File(test_dataframe,probability,prediction):
  df = test_dataframe
  probability_array = [x[1] for x in probability]
  df['off_prob'] = probability_array
        
  df['prediction'] = prediction

  #output the file
  df.to_csv(current_dir_path+'\\'+'MLP_test_model_results'+'.tsv',sep="\t",index=False)

# Create feature vector for the words.
def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    # This is a list that contains the names of the words in 
    # the model's vocabulary. 
    word_list = set(model.wv.index_to_key)

    # Loop over each word in the tweet and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in word_list: 
            nwords = nwords + 1
            featureVec = np.add(featureVec, model.wv[word])
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec

# This method is used to calculate the average of vectors.
def getVectorsAverage(tweets, model, num_features):
    # Given a set of tweets (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    tweetFeatureVecs = np.zeros((len(tweets), num_features), dtype="float32")
    counter = 0
    # Loop through the reviews
    for tweet in tweets:
        # Call the function to calculate the average of feature vectors
        tweetFeatureVecs[counter] = makeFeatureVec(tweet, model, num_features)
        counter = counter + 1
    return tweetFeatureVecs

# `train_MLP_model` method to trains a multi-layer perceptron model
# on the training data and returns that trained model.
#-------------------------------------------------------------------------------------------------
# @parameters - 
#   `path_to_train_file`: path of file which can be used for
#                        training the MLP model, input file could be olid(train_file),
#   `num_layers`: Number of layers use to train an MLP model on the entire dataset.
#-------------------------------------------------------------------------------------------------
def train_MLP_model(path_to_train_file, num_layers=2):
    # Read file one
    train_file = pd.read_csv(path_to_train_file, delimiter = '\t' )

    # Preprocessing file one.
    train_file['tweet'] = train_file['tweet'].apply(lambda x: preprocessed_data(x))

    # Extract pre-processed tweet values to train MLP model.
    preprocessed_tweets = train_file['tweet']

    # Train a word2vec model using training tweets.
    model = word2vec.Word2Vec(preprocessed_tweets,
        vector_size=300
    )

    # Training dataset average vectors in matrix form.
    avg_tweets = getVectorsAverage(preprocessed_tweets, model, 300)

    y = train_file['subtask_a'].values

    if num_layers == 1:
        layer_size = (50)
    elif num_layers == 3:
        layer_size = (30,50,50)
    else:
        layer_size = (30,30)

    clf = MLPClassifier(hidden_layer_sizes=layer_size,max_iter=1000)
    clf.fit(avg_tweets, y)
 
    return clf

# `test_MLP_model` method to test a multi-layer perceptron model
# on single, two and three layers.
# It provides the classification report and accuracy of the MLP model.
# This also outputs a test file (i.e. MLP_test_model_results.tsv)
# with probability and prediction labels.
#-------------------------------------------------------------------------------------------------
# @parameters - 
#   `path_to_test_file`: path of file which can be used for
#                        testing the trained MLP model, test file could be testset-levela(test_file),
#   `MLP_model`: This is the trained MLP model on desired number of layers.
#-------------------------------------------------------------------------------------------------
def test_MLP_model(path_to_test_file, MLP_model):
    #get the data file and create a pandas dataframe
    test_file_data = pd.read_csv(path_to_test_file, delimiter = '\t' )

    #clean the data using the function preprocessed_data()
    test_file_data['tweet'] = test_file_data['tweet'].apply(lambda x: preprocessed_data(x))

    #get the full labels dataset
    labels_a = current_dir_path+r'\labels-levela.csv'
    labels_df_full = pd.read_csv(labels_a,names = ['id','subtask_a'],header = None)

    #Extract the test data
    preprocessed_tweets = test_file_data['tweet']

    # Train Word2Vec model.
    model = word2vec.Word2Vec(preprocessed_tweets,
        vector_size=300
    )

    avg_tweets = getVectorsAverage(preprocessed_tweets, model, 300)

    # extract the classifications
    y_test = labels_df_full['subtask_a'].values

    # get the data from the classifier
    probabilty, prediction,score, report = evaluate_model(MLP_model,avg_tweets,y_test) 

    # Create the output test file with probability and prediction labels
    format_and_output_test_File(test_file_data,probabilty,prediction)

    print(report)
    print('score:',score)
