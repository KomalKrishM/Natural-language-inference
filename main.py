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
    
# neural network model
MIN_FREQ = 2

TEXT.build_vocab(train_data, 
                 min_freq = MIN_FREQ,
                 vectors = "glove.840B.300d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)
print(vars(valid_iterator))

class NLIBiLSTM(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim,
                 hidden_dim,
                 n_lstm_layers,
                 n_fc_layers,
                 output_dim, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
                                
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.translation = nn.Linear(embedding_dim, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, 
                            hidden_dim, 
                            num_layers = n_lstm_layers, 
                            bidirectional = True, 
                            dropout=dropout if n_lstm_layers > 1 else 0)
        
        fc_dim = hidden_dim * 2
        
        fcs = [nn.Linear(fc_dim * 2, fc_dim * 2) for _ in range(n_fc_layers)]
        
        self.fcs = nn.ModuleList(fcs)
        
        self.fc_out = nn.Linear(fc_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, prem, hypo):

        prem_seq_len, batch_size = prem.shape
        hypo_seq_len, _ = hypo.shape
    
        embedded_prem = self.embedding(prem)
        embedded_hypo = self.embedding(hypo)
                
        translated_prem = F.relu(self.translation(embedded_prem))
        translated_hypo = F.relu(self.translation(embedded_hypo))
                
        outputs_prem, (hidden_prem, cell_prem) = self.lstm(translated_prem)
        outputs_hypo, (hidden_hypo, cell_hypo) = self.lstm(translated_hypo)
        
        hidden_prem = torch.cat((hidden_prem[-1], hidden_prem[-2]), dim=-1)
        hidden_hypo = torch.cat((hidden_hypo[-1], hidden_hypo[-2]), dim=-1)
     
        hidden = torch.cat((hidden_prem, hidden_hypo), dim=1)
            
        for fc in self.fcs:
            hidden = fc(hidden)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
        
        prediction = self.fc_out(hidden)
        return prediction

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
N_LSTM_LAYERS = 2
N_FC_LAYERS = 3
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
decay_rate = 0

model = NLIBiLSTM(INPUT_DIM,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  N_LSTM_LAYERS,
                  N_FC_LAYERS,
                  OUTPUT_DIM,
                  DROPOUT,
                  PAD_IDX).to(device)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

model.embedding.weight.requires_grad = False

optimizer = optim.Adam(model.parameters(),weight_decay = decay_rate)


criterion = nn.CrossEntropyLoss().to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    prediction = []


    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            prem = batch.premise
            hypo = batch.hypothesis
            labels = batch.label
                        
            predictions = model(prem, hypo)
            
            
            loss = criterion(predictions, labels)
                
            acc = categorical_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            prediction += predictions
            
        
    return prediction,epoch_loss / len(iterator), epoch_acc / len(iterator)


path = "./BiLSTM.pt"
model.load_state_dict(torch.load(path))

prediction,test_loss, test_acc = evaluate(model, test_iterator, criterion)
for i in range(0,len(prediction)):
  prediction[i] = torch.argmax(prediction[i],dim=0)
print(len(prediction))

out1 = open('deep_model.txt','w')
for i in prediction:
  if i==0:
    out1.write('entailment')
    out1.write('\n')
  elif i==1:
    out1.write('contradiction')
    out1.write('\n')
  elif i==2:
    out1.write('neutral')
    out1.write('\n')
print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')