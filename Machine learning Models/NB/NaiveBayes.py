######################
#### Note: for training the model the functions is taking around 58 seconds to return the trainded NM mode l######
######################
import pandas as pd 
from nltk import word_tokenize
import nltk 
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
import timeit
import os
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# using timeit to time the code
start_time = timeit.default_timer()

# get current directory path.
current_dir_path = os.getcwd()

#Define the Input files
train_file = current_dir_path+r'\archive\olid-training-v1.0.tsv'
test_file_a = current_dir_path+r'\archive\testset-levela.tsv'
labels_a = current_dir_path+r'\archive\labels-levela.csv'

#Define the output file directory
file_out_dir = current_dir_path


# `clean_data` method to preprocess the data.
# Takes the Tweet string as input, performs lower, and
# splits it on any character that is not in the list.
def clean_data(input_string):
	stop_words = set(stopwords.words("english"))
	stop_words.add('user')
	stop_words.add('vs.')
	stop_words.add('url')
	stop_words.add('re')
	output_list = [x.strip(" '") for x in re.split(r"[^a-z']",input_string.lower())if len(x) > 0 and x not in stop_words]
	lemmatized_output_list = [WordNetLemmatizer().lemmatize(x) for x in output_list]
	stemmed_output_list = [PorterStemmer().stem(x) for x in lemmatized_output_list]
	output_list_string = " ".join(output_list)
	return output_list_string

#Function to write the dataframe out to a tsv file
def output_tsv_file(dataframe_in,file_path,file_name):
    dataframe_in.to_csv(file_path+'\\'+file_name+'.tsv',sep="\t",index=False)	

#merge the test file with ethe labels file on column 'id'
def merge_test_files(path_to_test_file,path_to_labels_file):
    # get the test file and create a pandas dataframe
    df = pd.read_csv(path_to_test_file, delimiter='\t')

    #get the full labels dataset
    labels_df_full = pd.read_csv(path_to_labels_file,names = ['id','subtask_a'],header = None)

    #merge the test dataset and the labels dataset in order to filter the test subsets(offensive, not offensive)
    test_DF = pd.merge(df,labels_df_full,on ='id')

    return test_DF	

# `train_NB_Model` method, to train NaiveBayes model using training dataset.
# returns an nltk.NaiveBayes language model- 
# outputs global variable (all_words)
# all_words is used to create the feature set on the test file data
#---------------------------------------------------------------------------
#@parameters--`path_to_train_file`: path of file which is used for
#               training NB model, input file could be olid(train_file)
#---------------------------------------------------------------------------
def train_NB_model(path_to_train_file):

		# get the test file and create a pandas dataframe
		df = pd.read_csv(path_to_train_file, delimiter='\t')

        # create a list of tuples, each tuple is >>> ('tweet','subtask_a')	
		tuple_list = list(zip(df.tweet, df.subtask_a))
        
        #Cleans the input data
		cleaned_list = [(clean_data(x[0]),x[1]) for x in tuple_list]

		#create a 'bag of words' list of all the words in the dataset
		all_words_list = [word.lower() for words in cleaned_list for word in word_tokenize(words[0]) if len(word) > 1 ]

		#Find the most common Number(x) words in the all_words_list
		all_words_dist_freq = nltk.FreqDist(all_words_list)
		frequent_words_list = all_words_dist_freq.most_common(2000)

		# filter the words list on the most common
		global all_words
		all_words = set(x[0] for x in frequent_words_list)		
	
		# Create thhe train using for loop
		train_list = []
		for x in cleaned_list:
			working_dictionary = {}
			text = x[0]
			label = x[1]
			tokenized_text = word_tokenize(text)
			for word in all_words:
				if word in tokenized_text:
					working_dictionary[word] = True
				else:
					working_dictionary[word] = False
			train_list.append((working_dictionary,label))		
		

		#train the model
		model = nltk.NaiveBayesClassifier.train(train_list)
	

		return model

#`test_NB_model` mthod to test NaiveBayes model against the test dataset.
# For each instance in the test file, it assigns a probability of being offensive .
# The function then generates an output file as the same format as the test file
# and adds 2 new columns off_probability is the probablity of being offensive and prediction is the class prediction (OFF|NOT).
#----------------------------------------------------------------------------------
## @parameters - 
#   `path_to_test_file`: path of file which can be used for
#                        testing trained NB model, input file could be olid(test_file),
#   `NB_model`: The Naive Bayes model output from train_NB_model method
def test_NB_model(path_to_test_file, NB_model):
		# get the test file and create a pandas dataframe
		df = merge_test_files(path_to_test_file,labels_a)

		# get a list of the labels for accuracy testing
		label_list = df['subtask_a'].to_list()

		#Clean and tokenize wach tweet
		tweet_list = df['tweet'].to_list()
		cleaned_tweet_list = [clean_data(x) for x in tweet_list]
		tokenized_tweet_list = [word_tokenize(x) for x in cleaned_tweet_list]
		
		
		# loop through the tweets and get the probablities and predictions from the model
		prediction_output_list = []
		off_prob_output_list = []

		for x in range(len(tokenized_tweet_list)): 
			tweet = tokenized_tweet_list[x]
			working_dictionary ={}
			for word in all_words:
				if word in tweet:
					working_dictionary[word] = True
				else:
					working_dictionary[word] = False
			prediction = NB_model.classify(working_dictionary)
			probablity_off = NB_model.prob_classify(working_dictionary).prob('OFF')			
			prediction_output_list.append(prediction)
			off_prob_output_list.append(probablity_off)

		#zip the actual labels to the predicted ones for comparison
		label_report_list = list(zip(label_list,prediction_output_list))

		# do all the math for reporting purposes
		not_count_accurate = 0
		not_count_innacurate = 0
		off_count_accurate = 0
		off_count_innacurate = 0

		for x,y in label_report_list:
				if x == 'NOT':
					if x == y:
						not_count_accurate += 1
					else:
						not_count_innacurate += 1
				else:
					if x == y:
						off_count_accurate += 1

					else:
						off_count_innacurate += 1

		total_not_count = not_count_innacurate+not_count_accurate
		total_off_count = off_count_innacurate+off_count_accurate
		percent_not_accuracy = not_count_accurate/total_not_count
		percent_off_accuracy = off_count_accurate/total_off_count
		percent_not_innacurate = not_count_innacurate/total_not_count
		percent_off_innacurate = off_count_innacurate/total_off_count

		#print out classifier accuracy details
		report = ("% non_off_accurate={},% non_offensive_innacurate = {} ,% off_accurate = {}, % off_innacurate= {}")
		print(report.format(percent_not_accuracy,percent_not_innacurate,percent_off_accuracy,percent_off_innacurate))

		# add the 2 required columns to the dataset
		df['off_probability'] = off_prob_output_list
		df['prediction'] = prediction_output_list

		#define and write the required output file
		df_out = df.drop('subtask_a', axis = 1)
		output_tsv_file(df_out,file_out_dir,'NaiveBayes_output')
		

		#ouput a second file containing subtsk_a for reporting and analyzing purposes
		output_tsv_file(df,file_out_dir,'NaiveBayes_output_analyze')

#NB_Model = train_NB_model(train_file)	
#test_NB_model(test_file_a,NB_Model)	

#Get the total time to run this file---- testing only
elapsed = timeit.default_timer() - start_time
print("The time to run the module was {} seconds ".format(elapsed))
