import pandas as pd
import re
from nltk.corpus import stopwords
from nltk import TweetTokenizer, PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix 
import os

# get current directory path.
current_dir_path = os.getcwd()

# Define the Input files
train_file = current_dir_path+r'\archive\olid-training-v1.0.tsv'
test_file_a = current_dir_path+r'\archive\testset-levela.tsv'
labels_a = current_dir_path+r'\archive\labels-levela.csv'

# `clean_data` method to handle preprocessing of the data.
# Removal of punctuations, stopwords.
# Takes the Tweet string as input, performs lower, and
# splits it on any character that is not in the list.
def clean_data(input_string):
  # Remove commonly used words i.e., stopwords.
  stop_words = set(stopwords.words("english"))
  stop_words.add('user')
  stop_words.add('vs.')
  stop_words.add('url')
  stop_words.add('re')

  # Keep stopwords like `i`, `not`, `user` which adds more sense in the data.
  stop_words.remove('i')
  stop_words.remove('not')
        
  out_list = [x.strip(" '") for x in re.split(r"[^a-z'.]",input_string.lower())if len(x) > 0 and x not in stop_words]
  out_string = " ".join(out_list)
  return out_string

# Another way to clean the data to try to improve the accuracy of the LR model.
def clean_data_version_2(input_string):
  #regrex_pattern = re.compile(pattern = "["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"  u"\U0001F1E0-\U0001F1FF" "]+", flags = re.UNICODE)
  #input_string = regrex_pattern.sub(r'',input_string)
  input_string = re.sub(r'\$\w*', '', input_string)
  input_string = re.sub(r'https?:\/\/.*[\r\n]*', '', input_string)
  input_string = re.sub(r'#', '', input_string)
  tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
  tweet_tokens = tokenizer.tokenize(input_string)
  stopwords_english = stopwords.words('english')
  stemmer = PorterStemmer()
  tweets_stem = []
  for word in tweet_tokens:
    if(word not in stopwords_english and word not in string.punctuation):
      tweets_stem.append(word)
      stem_word = stemmer.stem(word)
      tweets_stem.append(stem_word)
  return "".join(tweets_stem)

# `evaluate_model` method to handle probability, prediction of each test texts,
# calculate score and create report.
# It returns the prediction, score and report.
def evaluate_model(model,x,y):
  p_pred = model.predict_proba(x)
  y_pred = model.predict(x)
  score_ = model.score(x,y)
  conf_m = confusion_matrix(y,y_pred)
  report = classification_report(y,y_pred)
  return p_pred,y_pred,score_,report

# Function to extract Features
def vectorize_data(data_in):
  vectorizer = CountVectorizer(analyzer = 'word',ngram_range=(1,2), max_features=100)
  
  return vectorizer.fit_transform(data_in)

# Replace `CountVectorizer` with TfidfVectorizer`to compare the accuracy.
def vectorize_data_Tfidf(data_in):
  vectorizer = TfidfVectorizer(analyzer='word',
                            ngram_range=(1, 2),
                            max_features=100)
  return vectorizer.fit_transform(data_in)        

# Method to create logistic regression model.
# Returns the classifier model.
def train_model(x,y):
  #Classify the data
  classifier = LogisticRegression(max_iter=200, C=0.1, solver='liblinear')
  classifier.fit(x,y)
  return classifier

# `train_LR_model` method, to train LR model using training dataset.
# This method trains a Logistic Regression classification model.
# It returns trainded LR_model.
# --------------------------------------------------------------
#  @parameter - `path_to_train_file`: path of file which can be used for
#               training LR model
# ---------------------------------------------------------------
def train_LR_model(path_to_train_file):	
  #get the file and create a pandas dataframe
  df = pd.read_csv(path_to_train_file, delimiter = '\t' )

  #clean the data using the function clean_data()
  df['tweet'] = df['tweet'].apply(lambda x: clean_data_version_2(x))

  #extract the train data
  x_data = df['tweet'].values

  y_data = df['subtask_a'].values

  #vectorize the data
  x_train = vectorize_data(x_data)
  print(type(x_train))
  print(type(y_data))
  #create the LR model
  classifier = train_model(x_train,y_data)
  global model
  model =  classifier
  return model

# helper function to format and output the file for method test_LR_model
def format_and_output_test_File(test_dataframe,probability,prediction):
  df = test_dataframe
  probability_array = [x[1] for x in probability]
  df['off_prob'] = probability_array
        
  df['prediction'] = prediction

  #output the file
  df.to_csv(current_dir_path+'\\'+'log_reg_test_results'+'.tsv',sep="\t",index=False)

# `test_LR_model` method test the trained LR model against the test dataset.
# For each instance in the test file, 
# it calculates the class probablilty of the text ('OFF'|'NOT')
# The function then generates an output file as the same format as the test file
# and adds 2 new columns : offensive probability ('off_prob') and class prediction ('prediction').
# Prints a model evaluation report and score.
#-------------------------------------------------------------------------------------------------
# @parameters - 
#   `path_to_test_file`: path of file which can be used for
#                        testing trained LM model, input file could be olid(test_file),
#   `LR_model`: LR_model object created using train_LR_model method
#-------------------------------------------------------------------------------------------------
def test_LR_model(path_to_test_file, LR_model):
  #get the data file and create a pandas dataframe
  df = pd.read_csv(path_to_test_file, delimiter = '\t' )

  #clean the data using the function clean_data()
  df['tweet'] = df['tweet'].apply(lambda x: clean_data_version_2(x))

  #get the full labels dataset
  labels_df_full = pd.read_csv(labels_a,names = ['id','subtask_a'],header = None)

  #Extract the test data
  x_data = df['tweet'].values

  #Feature Extraction
  x_test = vectorize_data(x_data)

  # extract the classifications
  y_test = labels_df_full['subtask_a'].values

  #get the data from the classifier
  probabilty, prediction,score, report = evaluate_model(LR_model,x_test,y_test) 

  #Create the output file
  format_and_output_test_File(df,probabilty,prediction)

  print(report)
  print('score:',score)

def run():
  train_LR_model(train_file)        	
  test_LR_model(test_file_a,model)
