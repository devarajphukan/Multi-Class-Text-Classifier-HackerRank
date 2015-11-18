import pandas as pd
import numpy as np
import re, nltk        
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import *

#Reading the train and test data
train_data_df = pd.read_csv('trainingdata.txt',delimiter='\t',engine='python')
test_data_df = pd.read_csv('testingdata.txt',header = None ,delimiter="\n")

#Naming the columns in train and test set
train_data_df.columns = ["Domain","Text"]
test_data_df.columns = ["Text"]

#Using Porter Stemmer
stemmer = PorterStemmer()

#function to stem text tokens
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#function to tokenize text elements and removing numbers and punctuation
def tokenize(text):
    
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = text.split(" ")
    stems = stem_tokens(tokens, stemmer)
    return stems

# Tokenizing and vectorizing the text elements, eleminating stop words, using maximum of 1100 text tokens per document.
vectorizer = TfidfVectorizer(analyzer='word',tokenizer=tokenize,lowercase=True,stop_words ='english',max_features =1100)
corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())

#Convering the document term matrix to numpy nd array
corpus_data_features_nd = (corpus_data_features.toarray())
print corpus_data_features_nd.shape

# Model Declaration
# L2 regularization with hinged as the loss function and amount of regularization 0.7
my_model = LinearSVC(penalty = 'l2',dual = True,C=0.7,loss='hinge') 
#my_model = KNeighborsClassifier()

# Fit model with train and test data
my_model = my_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Domain)
test_pred = my_model.predict(corpus_data_features_nd[len(train_data_df):])

spl = []
for i in range(len(test_pred)) :
    spl.append(i)
results = []
actual = []

#File containing all the real output class
foput = open("actual_output.txt","r")
for m in foput :
    m = str(m).strip()
    actual.append(int(m))

# Getting Prediction Results
for text, Domain in zip(test_data_df.Text[spl], test_pred[spl]):
    #print Domain,"\n"
    a = str(Domain).strip()
    results.append(int(a))

# Comparing Results with actual classes
correct = 0
wrong = 0
for i in range(0,len(actual)) :
    if results[i] == actual[i] :
        correct += 1
    else :
        wrong += 1

#Getting accuracy
accuracy = 100 * (float((correct-wrong))/len(actual))
print "accuracy : ",accuracy