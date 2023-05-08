## Introduction

Implemented few natural language processing tasks to undertstand and gain the knowledge about natural language processing applications such as speech recognition, machine translation, text generation, text summarization and language modelling. 

Broadly, it is classified into three tasks , machine learning models, machine learning models with Neural Networks, and Deep learning models. 

### Machine Learning Models:

To understand the basic machine learning models, created LM tasks, analyzed the dataset using language mode and implemented logistic regression to classify the same dataset.  

#### 1. Dataset:

The dataset that will be used is the OLID dataset.
Note: For the LM and LR tasks only the the first label (Offensive or Nonoffensive)is considered.

The training set (olid-training-v1.0.tsv) is used for training the LM and LR, while the test set (testset-levela.tsv) and labels (labels-levela.csv) is used for testing and analyzing your models.

#### 2. Language Model Tasks:

A language model is a type of statistical model that is used to predict the probability of a sequence of words in a language. 

Two functions for LMs:

##### train_LM(path_to_train_file)

This method trains up to a bigram LM and returns the trained LM. The format for the train file should follow the same format as the olid-training data file!

##### test_LM(path_to_test_file, LM_model)

This method tests an LM on a test file. That is, for each instance in the test file, it assigns a MLE score (refer to in class notes for how we approach this) for each test text. The function then generates an output file as the same format as the test file but adds a new column with the MLE score. The format for the input file should follow that of the olid test file.
Once these functions are implemented use them to accomplish the following tasks.

1. Create 3 LMs:
a. LM_full -> trained on all the olid training data.
b. LM_not -> trained on all the non-offensive (NOT) training data
c. LM_off -> trained on all the offensive (OFF) training data
2. For each LM, test on: 1) the full test set, 2) the not offensive subset, 3) the offensive subset
3. For each test file, make observations (e.g. averages of scores for LM_full on the offensive subset compared to LM_not) and analyze if your LMs seem to be successfully capturing the language they were trained on.
4. Make a observational report based on the analysis.

#### 3. Logisitic Regression Models Tasks:

Two functions for LR:

##### train_LR_model(path_to_train_file)

This method trains a logistic regression model on some set of features and returns that trained model. The format for the train file should follow the same format as the olid-training data file!

##### test_LR_model(path_to_test_file, LR_model)

This method tests a trained LR model on some test file and outputs a test file in the same format as the input test file but with 2 columns added: 1) probability of that text being offensive, 2) class prediction (OFF, NOT). The format for the input file should follow that of the olid test file.
Once these functions are implemented use them to accomplish the following tasks.

1. Train a LR model on the entire train set.
2. Test the trained model on the test set and produce predictions for all test texts.
3. Analyze the accuracy of the LR model and try to improve it. This may be done by: adding other features in, improving your preprocessing of the text, changing the hyperparameters of the LR model for training, etc.
4. Create a analysis report

#### 4. Naive Bayes Model Tasks:

##### train_NB_model(path_to_train_file)

This method trains a naïve bayes model on the training text and returns that trained model. The format for the train file should follow the same format as the olid-training data file!

##### test_NB_model(path_to_test_file, NB_model)

This method tests a trained NB model on some test file and outputs a test file in the same format as the input test file but with 2 columns added: 1) probability of that text being offensive, 2) class prediction (OFF, NOT). The format for the input file should follow that of the olid test file.

Once these functions are implemented accomplish the following tasks.
1. Train a NB model on the entire train set.
2. Test the trained model on the test set and produce predictions for all test texts.
3. Analyze your accuracy of your NB model and compare to the LR model. Compare and contrast how the models did on different labeled data (ie NOT vs OFF texts).
4. Write the analysis and comparisons to the report file.

### Machine Learning Models with word2Vec and Neural Networks:

Implemented word2vec embeddings to understand them and then incorporated them with simple neural networks for classification.

#### 1. Dataset:

Same OLID dataset has been used for this task as well.

#### 2. Word2Vec Tasks:

The Word2Vec algorithm consists of two main models: Continuous Bag of Words (CBOW) and Skip-gram. CBOW predicts a target word based on its surrounding context words, while Skip-gram predicts context words based on a target word. Both models use a neural network to train the word embeddings.

##### compare_texts_word2vec(file_one, file_two, k = 10)

This method takes in two dataset files and compares them at the word level by leveraging word2vec. First, this function should find the k most common non stop words for each file.
Then these 2*k words will be used to calculate 2 things:

1) Similarity between the 2 text files based on the top k words. It is up to you to determine how you want to calculate this similarity score, but cosine simlarity should be involved and the result should be a numerical score. (You should also document what algorithm you use).

2) Unique words for file_one, file_two and overlapping words for the two files. Rather than limiting to the 2*k words, you should additionally obtain the 10 most similar words for each k word leveraging word2vec to do so. This <= 10 *k words will then be used to find the required lists of words.

With the above function, perform the following function calls:

1. compare_texts_word2vec(NOT_subset, OFF_subset, k = 5)
2. compare_texts_word2vec(NOT_subset, OFF_subset, k = 10)
3. compare_texts_word2vec(NOT_subset, OFF_subset, k = 20)

The above statistics is mentioned in the final report document. Additionally, also made some observations about both the similarity statistics and the words.

#### 3. Multi-Layer Perceptron Tasks:

It is a type of neural network that consists of multiple layers of interconnected neurons. An MLP is typically used for supervised learning tasks such as classification and regression.

##### train_MLP_model(path_to_train_file, num_layers = 2)

This method trains a multi-layer perceptron model on some training data and returns that trained model. The training texts should be represented by word2vec embeddings. You may use any pretrained word2vec embeddings you choose. Recall that the input size will affect how much of the input text is able to be sent in to the model. The MLP slides had some possible solutions, so you may choose any of these (but you should always note your decisions). The format for the train file should follow the same format as the olid-training data file!

##### test_MLP_model(path_to_test_file, MLP_model)

This method tests a trained MLP model on some test file and outputs a test file in the same format as the input test file but with 2 columns added: 1) probability of that text being offensive, 2) class prediction (OFF, NOT). The format for the input file should follow that of the olid test file.
Once these functions are implemented, accomplish the following tasks.
1. Train a 2 layer MLP model on the entire train set.
2. Test the trained model on the test set and produce predictions for all test texts.
3. Repeat 1 and 2 for a 1 layer MLP and a 3 layer MLP. Compare and contrast the overall accuracies for the models.
4. Create the observational report.

#### 3. Improving Pretrained word2vec embeddings:

##### update_embeddings(path_to_train_file)

This method fine-tunes a pretrained word2vec embedding (your choice) on the training text and then saves the fine-tuned embeddings. The function should then return the path to the fine-tuned
embeddings. The format for the train file should follow the same format as the olid-training data file!

Modify your MLP function to take in an argument for the word2vec embeddings to use.

Once these functions are implemented and modified, accomplish the following tasks.

1. Train one set of pretrained word2vec embeddings on the OLID training text.
2. Test one of your trained MLP models on the test set producing scores for both the original pretrained embeddings and the fine-tuned embeddings. (Note that you will need to train one MLP on the fine-tuned and one on the pretrained.)
3. Compare the overall accuracies of the pretrained embeddings MLP to the fine-tuned embeddings MLP. Are the accuracies better, worse, or the same? Is this surprising or not? What might be causing the results?
4. Write the analysis and comparisons to your report file.

### Deep Learning Models:

Pretrained transformer models which can be useful for baseline comparison or even finetuning however, focus on a similar task (e.g. sentiment classification IMDB, sarcasm detection, AG News classification).

The comparison should involve using a built in hugging face dataset and testing the accuracies on at least 100 examples of the test set portion of that dataset. Comparison were done between at least 3 different models.

#### 1. Pre-trained transformer models:

1. Choose pretrained models to evaluate
a. Choose 3 models which are all trained in the same classification task
OR
b. Choose 4 models, 2 trained on same task A and 2 trained on same task B
2. Choose relevant dataset to compare pretrained models on (each dataset should use at least
100 examples for comparison)
3. Write a simple report:
a. Summarize accuracy of all models for their corresponding datasets
b. Write out explicit comparisons and discussion of results (were there any surprises?
Any possible bugs that could be causing problems?)
c. Small section on limitations of your comparisons
d. Small section on how you could use these pretrained models in the future if you were
to do research in these directions

#### 2. Fine-tune a Bert based model:

To fine-tune transformer and Pytorch has been used.

Following tasks for the fine-tuning the bert base model:
1. Using transformers and pytorch, fine-tune a bert-based-uncased model on the subtask A of
the OLID dataset (OFF or NOT).
2. Produce predictions for the test portion of OLID.
3. Calculate accuracy.
4. In a SEPARATE report, report the results and the previous “simpler” models’ results from
previous basic, neural network machine learning models and deep learningmodels.
5. Compare and contrast the results for the models and provide explicit report.





