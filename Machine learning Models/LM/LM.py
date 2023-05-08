import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.lm.preprocessing import padded_everygram_pipeline, flatten
from nltk.lm import MLE
from nltk import bigrams
import os
import re
import math

#import nltk # only need this for the data download
#nltk.download("stopwords") #Only need to run this once
#nltk.download('punkt') #Only need to run this once

# get current directory path.
current_dir_path = os.getcwd()

# Define the Input files
train_file = current_dir_path+r'\archive\olid-training-v1.0.tsv'
test_file_a = current_dir_path+r'\archive\testset-levela.tsv'
labels_a = current_dir_path+r'\archive\labels-levela.csv'
offensive_train_file = current_dir_path+r'\offensive_train_file.tsv'
not_offensive_train_file = current_dir_path+r'\not_offensive_train_file.tsv'

#Function to write the dataframe out to a tsv file
def output_tsv_file(dataframe_in,file_path,file_name):
    dataframe_in.to_csv(file_path+'\\'+file_name+'.tsv',sep="\t",index=False)

def create_offensive_train_file():
    df = pd.read_csv(train_file, delimiter = '\t' )
    working_DF = df[df['subtask_a'] == 'OFF']
    output_tsv_file(working_DF,current_dir_path,'offensive_train_file')
    print("Your offensive data train file was saved as {}".format(current_dir_path+'\\offensive_train_file.tsv'))

def create_not_offensive_train_file():
    df = pd.read_csv(train_file, delimiter = '\t' )
    working_DF = df[df['subtask_a'] == 'NOT']
    output_tsv_file(working_DF,current_dir_path,'not_offensive_train_file')
    print("Your non-offensive data train file was saved as {}".format(current_dir_path+'\\not_offensive_train_file.tsv'))

def merge_test_files():
    # get the test file and create a pandas dataframe
    df = pd.read_csv(test_file_a, delimiter='\t')

    #get the full labels dataset
    labels_df_full = pd.read_csv(labels_a,names = ['id','subtask_a'],header = None)

    #merge the test dataset and the labels dataset in order to filter the test subsets(offensive, not offensive)
    test_DF = pd.merge(df,labels_df_full,on ='id')

    return test_DF        

# Methods to handle creation of training files containing
# non-offensive and offensive data separately.
# Note: If the code LM.py is executed again, do not forget to close the files,
# otherwise you will get a permission error.
create_not_offensive_train_file()
create_offensive_train_file()

# `clean_data` method to handle preprocessing of the data.
# Removal of punctuations, stopwords.
# Takes the Tweet string as input, performs lower, and
# splits it on any character that is not in the list.
def clean_data(input_string):
    # Remove commonly used words i.e., stopwords. 
    stop_words = set(stopwords.words("english"))
    stop_words.add('vs.')
    stop_words.add('url')
    stop_words.add('re')
    stop_words.add('user')

    # Keep stopwords like `i`, `not`, `user` which adds more sense in the data.
    stop_words.remove('i')
    stop_words.remove('not')

    cleaned_data = [x.strip(" '") for x in re.split(r"[^a-z'.]",input_string.lower())if len(x) > 0 and x not in stop_words]
    #cleaned_data leaves the end of sentence period attached to the word.
    cleaned_data_string = " ".join(cleaned_data)
    cleaned_data = word_tokenize(cleaned_data_string)
    return cleaned_data

# `train_LM` method, to train LM model using training dataset.
# This method train a Maximum Likelihood Estimator(MLE) for bigrams.
# Returns following three LM models based on the type of input training file:
#   a. LM_full -> trained on all the olid training data.
#   b. LM_not  -> trained on all the non-offensive (NOT) training data
#   c. LM_off  -> trained on all the offensive (OFF) training data
#------------------------------------------------------------------------
# @parameter - `path_to_train_file`: path of file which can be used for
#               training LM model, input file could be olid(train_file),
#               non-offensive(offensive_train_file), or,
#               offensive(not_offensive_train_file) training data.
#------------------------------------------------------------------------
def train_LM(path_to_train_file):

    #get the file and create a pandas dataframe
    #--Note--# a dataframe is easier to manipulate compared to multiple "for" loops
    df = pd.read_csv(path_to_train_file, delimiter = '\t' )

    # Training data preprocessing using `clean_data` method.
    df['bigrams'] = df['tweet'].apply(lambda x: clean_data(x))

    # Create a list of tweets to create the model
    tweet_list = df['bigrams'].tolist()

    # Create the train ngram and vocab string for the model
    train, vocab = padded_everygram_pipeline(2,tweet_list)              

    # Create the language model
    lm = MLE(2)

    # Fit the model
    lm.fit(train,vocab)
    return lm

#`test_LM` method test every three trained LM models against each test dataset.
# For each instance in the test file, it assigns a MLE score for each test text.
# The function then generates an output file as the same format as the test file
# and adds a new column with the MLE score.
# For each test file, calculate avergae score against each LM models.
#------------------------------------------------------------------------
# @parameters - 
#   `path_to_test_file`: path of file which can be used for
#                        testing trained LM model, input file could be olid(test_file),
#                        non-offensive(offensive_test_file), or,
#                        offensive(not_offensive_test_file) test data.
#   `LM_model`: For each type of LM model(i.e., `LM_full`, `LM_not`, and, `LM_off`), this method test on
#                   a. the full test set
#                   b. the not offensive subset,
#                   c. the offensive subset
#------------------------------------------------------------------------
def test_LM(path_to_test_file,LM_model):

    # get the test file and create a pandas dataframe
    df = pd.read_csv(path_to_test_file, delimiter='\t')

    # get the full labels dataset
    labels_df_full = pd.read_csv(labels_a,names = ['id','subtask_a'],header = None)

    # merge the test dataset and the labels dataset in order to
    # filter the test subsets(offensive, not offensive).
    test_DF = pd.merge(df,labels_df_full,on ='id')

    # Test data preprocessing using `clean_data` method.
    # Split the tweet into words list the same as we did the train dataset.
    # Takes the tweet column and applies the normalization and tokenization
    # returns a new column "clean_data".
    test_DF['clean_data'] = test_DF['tweet'].apply(lambda x: list(flatten(clean_data(x))))

    # Create the "bigrams" column from "clean_data" column.
    # Takes the formatted list and turns it into bigrams to feed the model.
    test_DF['bigrams'] = test_DF['clean_data'].apply(lambda x: list(bigrams(x)))
    
    # Full test set MLE Score.
    # Finds the MLE scores of each bigram in the tweet.
    test_DF['MLE_data'] = test_DF['bigrams'].apply(lambda x: [LM_model.score(y[1],y[0].split()) for y in x] )

    # Smoothing: Substitute a 1 for any score of 0
    test_DF['MLE_data_filtered'] = test_DF['MLE_data'].apply(lambda x: [y if y != 0 else 1 for y in x] )

    # computes the score of the text by taking product of the individual bigram scores(Multiply).
    test_DF['MLE_score'] = test_DF['MLE_data_filtered'].apply(lambda x: math.prod(x))

    # save score of each test data against each model in a csv file at this location.
    score_file_dir = current_dir_path+r'\LM_testData_score'

    # output the full test data file.
    full_df_out = test_DF[['id','tweet','MLE_score']]
    output_tsv_file(full_df_out,score_file_dir,'full_test_data_output')

    # calculate average score for full.
    print("-------------------------")
    print('Average score - full_TestData: ', full_df_out['MLE_score'].mean())
    print("Your full data test results file was saved as {}".format(score_file_dir+'\\full_test_data_output.tsv'))
    print("-------------------------")

    # filter the Not offensive test data and output to file.
    not_df = test_DF[test_DF['subtask_a'] == 'NOT']
    not_df_out = not_df[['id','tweet','MLE_score']]
    output_tsv_file(not_df_out,score_file_dir,'not_off_test_output')

    # calculate average score for non-offensive.
    print('Average score - not_off_TestData: ', not_df_out['MLE_score'].mean())
    print("Your non-offensive data test results file was saved as {}".format(score_file_dir+'\\not_off_test_output.tsv'))
    print("-------------------------")

    # filter the Offensive output test data and output to file.
    off_df = test_DF[test_DF['subtask_a'] == 'OFF']
    off_df_out = off_df[['id','tweet','MLE_score']]
    output_tsv_file(off_df_out,score_file_dir,'off_test_data_output')

    # calculate avergae score for offensive.
    print('Average score - off_TestData: ', off_df_out['MLE_score'].mean())
    print("Your offensive data test results file was saved as {}".format(score_file_dir+'\\off_test_data_output.tsv'))
    print("-------------------------")

# For ease of testing,
# I have included these functions too
# beside the two required functions(i.e., `train_LM`, `test_LM`).
# run this function after you load the file.
def run():
   lm_model = train_LM(train_file)
   test_LM(test_file_a,lm_model)
print('done')
