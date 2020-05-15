import os
import requests
cwd = os.getcwd()

url =  "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
print(cwd)
target_path = cwd + '/snli_1.0.zip'
print(target_path)

response = requests.get(url, stream=True)
handle = open(target_path, "wb")
for chunk in response.iter_content(chunk_size=512):
    if chunk:  # filter out keep-alive new chunks
        handle.write(chunk)
handle.close()

from zipfile import ZipFile
# specifying the zip file name
file_name = "snli_1.0.zip"
dest_path = cwd + '\.data\snli'

with ZipFile(file_name, 'r') as zipobj:
    listOfFileNames = zipobj.namelist()
    print(listOfFileNames)
    for fileName in listOfFileNames:
        if fileName.endswith('.jsonl'):
           # Extract a single file from zip
           print('Extracting')
           zipobj.extract(fileName, dest_path)
           print('Done!')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import random
import numpy as np

import re

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(sequential = False, lower = True)
LABEL = data.LabelField()

train_data, valid_data, test_data = datasets.SNLI.splits(TEXT, LABEL)

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")


valid_dataset = []
valid_labels = []
valid_corpus = []

train_dataset = []
train_labels = []
train_corpus = []

test_dataset = []
test_labels = []
test_corpus = []

def trainpreprocessing(length,pro_data):
  for i in range(0,length):
    line = vars(pro_data.examples[i])
    data = list(line.items())
    train_dataset.append(data[0][1]+data[1][1])
    train_labels.append(data[2][1])
    train_dataset[i] = re.sub('[^a-zA-Z]',' ',train_dataset[i])
    train_corpus.append(train_dataset[i])
    if train_labels[i]=='neutral':
          train_labels[i]=0
    elif train_labels[i]=='entailment':
        train_labels[i]=1
    elif train_labels[i]=='contradiction':
        train_labels[i]=2
  return train_corpus,train_labels

def validpreprocessing(length,pro_data):
  for i in range(0,length):
    line = vars(pro_data.examples[i])
    data = list(line.items())
    valid_dataset.append(data[0][1]+data[1][1])
    valid_labels.append(data[2][1])
    valid_dataset[i] = re.sub('[^a-zA-Z]',' ',valid_dataset[i])
    valid_corpus.append(valid_dataset[i])
    if valid_labels[i]=='neutral':
      valid_labels[i]=0
    elif valid_labels[i]=='entailment':
      valid_labels[i]=1
    elif valid_labels[i]=='contradiction':
      valid_labels[i]=2
  return valid_corpus,valid_labels

def testpreprocessing(length,pro_data):
  for i in range(0,length):
    line = vars(pro_data.examples[i])
    data = list(line.items())
    test_dataset.append(data[0][1]+data[1][1])
    test_labels.append(data[2][1])
    test_dataset[i] = re.sub('[^a-zA-Z]',' ',test_dataset[i])
    test_corpus.append(test_dataset[i])
    if test_labels[i]=='neutral':
          test_labels[i]=0
    elif test_labels[i]=='entailment':
        test_labels[i]=1
    elif test_labels[i]=='contradiction':
        test_labels[i]=2
  return test_corpus,test_labels

train_corpus,train_labels = trainpreprocessing(len(train_data),train_data)

valid_corpus,valid_labels = validpreprocessing(len(valid_data),valid_data)

test_corpus,test_labels = testpreprocessing(len(test_data),test_data)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#import joblib

tf = TfidfVectorizer()
x = tf.fit_transform(train_corpus)
logistic_model = LogisticRegression(C=15,max_iter=10000)
logistic_model.fit(x,train_labels)

x_test = tf.transform(test_corpus)
y_pred = logistic_model.predict(x_test)

print('\n logistic_training_accuracy:',accuracy_score(test_labels,y_pred))

print(y_pred)

# Save to file in the current working directory
#joblib_file = "joblib_model.pkl"
#joblib.dump(model, joblib_file)
#
#test_dataset = []
#test_labels = []
#test_corpus = []

# Load from file
#joblib_model = joblib.load(joblib_file)
#
#
#def testpreprocessing(length,pro_data):
#  for i in range(0,length):
#    line = vars(pro_data.examples[i])
#    data = list(line.items())
#    test_dataset.append(data[0][1]+data[1][1])
#    test_labels.append(data[2][1])
#  #test_dataset = re.sub('[^a-zA-Z]',' ',test_dataset)
#    if test_labels[i]=='neutral':
#          test_labels[i]=0
#    elif test_labels[i]=='entailment':
#        test_labels[i]=1
#    elif test_labels[i]=='contradiction':
#        test_labels[i]=2
#  return test_dataset,test_labels
#
#test_corpus,test_labels = testpreprocessing(len(test_data),test_data)

# Calculate the accuracy and predictions
#score = joblib_model.score(Xtest, Ytest)
##print("Test score: {0:.2f} %".format(100 * score))
#x_test = tf.transform(test_corpus)
#y_predict = joblib_model.predict(x_test)
#
#print('\n logistic_test_accuracy:',accuracy_score(test_labels,y_predict))

out = open('tfidf.txt','w')

for i in y_pred:
  if i==0:
    out.write('entailment')
    out.write('\n')
  elif i==1:
    out.write('contradiction')
    out.write('\n')
  elif i==2:
    out.write('neutral')
    out.write('\n') 

