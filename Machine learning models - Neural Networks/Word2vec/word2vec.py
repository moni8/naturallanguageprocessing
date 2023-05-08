## Word2Vec - Homework #3 - MoniKa Vyas CS 59000 14 NLP
import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords
import math
import collections
import gensim.downloader as api

#nltk.download("stopwords") #Only need to run this once
#nltk.download('punkt')

# Get the current working directory path
current_dir_path = os.getcwd()

twitter_dataset = current_dir_path+r'\olid-training-v1.0.tsv'

# Function to write the dataframe out to a tsv file
def output_tsv_file(dataframe_in,file_path,file_name):
    dataframe_in.to_csv(file_path+'\\'+file_name+'.tsv',sep="\t",index=False)

def create_not_offensive_subset_file():
    df = pd.read_csv(twitter_dataset, delimiter = '\t' )
    working_DF = df[df['subtask_a'] == 'NOT']
    output_tsv_file(working_DF,current_dir_path,'not_offensive_subset')
    print("Your non-offensive subset file was saved at {}".format(current_dir_path+'\\not_offensive_subset.tsv'))

def create_offensive_subset_file():
    df = pd.read_csv(twitter_dataset, delimiter = '\t' )
    working_DF = df[df['subtask_a'] == 'OFF']
    output_tsv_file(working_DF,current_dir_path,'offensive_subset')
    print("Your offensive subset file was saved at {}".format(current_dir_path+'\\offensive_subset.tsv'))

# Methods to handle creation of subset files containing
# non-offensive and offensive data separately.
create_not_offensive_subset_file()
create_offensive_subset_file()

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
    
    # preprocessed_data leaves the end of sentence period attached to the word.
    preprocess_data_string = " ".join(preprocessed_data)
    preprocessed_data = word_tokenize(preprocess_data_string)
    return preprocessed_data

# Method to handle reading and pre-processing of file.
# This function will pre-process the file data.
# Performing second level of stop words removal for better cosine similarity.
# This method returns the list of pre-processed words extracted from the file.
def read_preprocess_file(file):
    file_data = pd.read_csv(file, delimiter = '\t' )
    file_data['preprocessed_file_one_data'] = file_data['tweet'].apply(lambda x: preprocessed_data(x))
    file_merged_list = file_data['preprocessed_file_one_data'].explode().to_list()
    file_filtered = ' '.join(map(str, file_merged_list))
    file_filtered_string = remove_stopwords(file_filtered)
    file_data_to_list = list(file_filtered_string.split(" "))
    return file_data_to_list

# returns the dot product of two files
def dotProduct(file1, file2):
	sum = 0.0
	for key in file1:
		if key in file2:
			sum += (file1[key] * file2[key])
	return sum

# returns the cosine angle in radians
# between file vectors
def calculate_similarity(file1, file2):
    dotproduct = dotProduct(file1, file2)
    denominator = math.sqrt(dotProduct(file1, file1)*dotProduct(file2, file2))

    return math.acos(dotproduct / denominator)

# Method to find k-most common non-stop words in the given file.
# This method will return k-most common non-stop words in the form of dictionary.
def k_most_common_non_stopwords(file, k):
    list_of_words = read_preprocess_file(file)
    word_frequency = collections.Counter(list_of_words)
    k_most_common_words = word_frequency.most_common(k)
    k_most_dict = dict(k_most_common_words)
    return k_most_dict

# This method returns the 10 most similar words for each k word.
# Word2Vec pre-trained model used to find the new list of similar words.
def word2vec_similar_words(word_list):
    wv = api.load('word2vec-google-news-300')
    new_words = []
    for word in word_list:
        if word in wv:
            file_one_similar_words = wv.most_similar(positive= [word], topn = 10)
            new_words.extend(file_one_similar_words)
            #new_words.append(file_one_similar_words)
    new_words_dict = dict(new_words)
    return new_words_dict

# Method to handle word comparison between two text files using word2vec.
# First find the k most common non stop words for each file.
# This method finds out the similarity between two files,
# unique words and overlapping words from both the files.
#-------------------------------------------------------------------------------------------------
# @parameters - 
#   `file_one`: path of the file one which can be used to compare to second file for word2vec tasks.
#   `file_two`: path of the file two which can be used to compare to first file for word2vec tasks.
#   `k`: Number of most common non-stop words. Default: 10
#-------------------------------------------------------------------------------------------------
def compare_texts_word2vec(file_one, file_two, k=10):

    # Get k-most common non-stop words from both the files.
    file1_dict = k_most_common_non_stopwords(file_one, k)
    file2_dict = k_most_common_non_stopwords(file_two, k)

    # Calculate cosine similarity between two files using cosine similarity formula.
    similarity = calculate_similarity(file1_dict, file2_dict)
    print("The cosine similarity between the two files is: % 0.6f (radians)"% similarity,"\n")

    # Get 10 most similar words for each k word using word2vec.
    first_file_word2vec_words = word2vec_similar_words(file1_dict)
    second_file_word2vec_words = word2vec_similar_words(file2_dict)    

    # Unique words for both the files.
    file_one_unique_words = [word for word in first_file_word2vec_words if word not in second_file_word2vec_words]
    file_two_unique_words = [word for word in second_file_word2vec_words if word not in first_file_word2vec_words]
    print("File one Unique Word's List: ",file_one_unique_words,"\n")
    print("File two Unique Word's List: ",file_two_unique_words,"\n")

    # Overlapping words between both the files.
    overlapped_words = set(first_file_word2vec_words)&set(second_file_word2vec_words)
    print("Overlapped words between both the files are: ",overlapped_words)
