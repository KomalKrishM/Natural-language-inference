import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets

import random
import numpy as np

import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', lower = True)
LABEL = data.LabelField()
print(vars(TEXT))
print(vars(LABEL))

train_data, valid_data, test_data = datasets.SNLI.splits(TEXT, LABEL)

print(vars(valid_data.examples[10]))
print(vars(train_data.examples[10]))

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

print(vars(train_data.examples[0]))

MIN_FREQ = 2

TEXT.build_vocab(train_data, min_freq = MIN_FREQ,vectors = "glove.840B.300d",unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in TEXT vocabulary: {len(LABEL.vocab)}")
print(vars(LABEL.vocab))

print(TEXT.vocab.freqs.most_common(20))

print(TEXT.vocab.itos[:10])

print(LABEL.vocab.itos)
print(LABEL.vocab.stoi)

print(LABEL.vocab.freqs.most_common())

print(len(train_data))

BATCH_SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)
print(len(valid_iterator))
print(len(test_iterator))
print(len(train_iterator))
for batch in test_iterator:
  prem = batch.premise
  hypo = batch.hypothesis
  labels = batch.label
#  print(prem.shape)
print(batch)
print(prem)
print(labels)
print(prem.shape)
print(labels.shape)
print(len(prem))
print(len(labels))

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
        
        # prem = [prem sent len, batch size]
        # hypo = [hypo sent len, batch size]
        
        embedded_prem = self.embedding(prem)
        embedded_hypo = self.embedding(hypo)
        
        # embedded_prem = [prem sent len, batch size, embedding dim]
        # embedded_hypo = [hypo sent len, batch size, embedding dim]
        
        translated_prem = F.relu(self.translation(embedded_prem))
        translated_hypo = F.relu(self.translation(embedded_hypo))
        
        # translated_prem = [prem sent len, batch size, hidden dim]
        # translated_hypo = [hypo sent len, batch size, hidden dim]
        
        outputs_prem, (hidden_prem, cell_prem) = self.lstm(translated_prem)
        outputs_hypo, (hidden_hypo, cell_hypo) = self.lstm(translated_hypo)

        # outputs_x = [sent len, batch size, n directions * hid dim]
        # hidden_x = [n layers * n directions, batch size, hid dim]
        # cell_x = [n layers * n directions, batch size, hid dim]
        
        hidden_prem = torch.cat((hidden_prem[-1], hidden_prem[-2]), dim=-1)
        hidden_hypo = torch.cat((hidden_hypo[-1], hidden_hypo[-2]), dim=-1)
        
        # hidden_x = [batch size, fc dim]

        hidden = torch.cat((hidden_prem, hidden_hypo), dim=1)

        # hidden = [batch size, fc dim * 2]
            
        for fc in self.fcs:
            hidden = fc(hidden)
            hidden = F.relu(hidden)
            hidden = self.dropout(hidden)
        
        prediction = self.fc_out(hidden)
        
        # prediction = [batch size, output dim]
        
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)
print(pretrained_embeddings)

model.embedding.weight.data.copy_(pretrained_embeddings)

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

model.embedding.weight.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(),weight_decay = decay_rate)

criterion = nn.CrossEntropyLoss().to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        prem = batch.premise
        hypo = batch.hypothesis
        labels = batch.label
        
        optimizer.zero_grad()
        
        #prem = [prem sent len, batch size]
        #hypo = [hypo sent len, batch size]
        
        predictions = model(prem, hypo)
        
        #predictions = [batch size, output dim]
        #labels = [batch size]
        
        loss = criterion(predictions, labels)
                
        acc = categorical_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

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

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 9

#best_valid_loss = float('inf')

start_time = 0
end_time = 0

for epoch in range(N_EPOCHS):
    decay_rate = 0.001/(epoch+1)

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    prediction,valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

from google.colab import drive
drive.mount('/content/gdrive')
!ls /content/gdrive
model_save_name = 'BiLSTM_W_DECAY.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"
torch.save(model.state_dict(), path)

# path = F"/content/gdrive/My Drive/{BiLSTM_model.pt}"
# model.load_state_dict(torch.load('BiLSTM_model.pt'))

model_save_name = 'BiLSTM_W_DECAY.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"
model.load_state_dict(torch.load(path))

prediction,test_loss, test_acc = evaluate(model, test_iterator, criterion)
for i in range(0,len(prediction)):
  prediction[i] = torch.argmax(prediction[i],dim=0)
print(len(prediction))

print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

out1 = open('deep_model.txt','w')
print(len(prediction))

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

def predict_inference(premise, hypothesis, text_field, label_field, model, device):
    
    model.eval()
    
    if isinstance(premise, str):
        premise = text_field.tokenize(premise)
    
    if isinstance(hypothesis, str):
        hypothesis = text_field.tokenize(hypothesis)
    
    if text_field.lower:
        premise = [t.lower() for t in premise]
        hypothesis = [t.lower() for t in hypothesis]
        
    premise = [text_field.vocab.stoi[t] for t in premise]
    hypothesis = [text_field.vocab.stoi[t] for t in hypothesis]
    
    premise = torch.LongTensor(premise).unsqueeze(1).to(device)
    hypothesis = torch.LongTensor(hypothesis).unsqueeze(1).to(device)
    
    prediction = model(premise, hypothesis)
    
    prediction = prediction.argmax(dim=-1).item()
    
    return label_field.vocab.itos[prediction]

premise = 'a man sitting on a green bench.'
hypothesis = 'a woman sitting on a green bench.'

predict_inference(premise, hypothesis, TEXT, LABEL, model, device)
#print(label_field.vocab.itos[prediction])

premise = 'a man sitting on a green bench.'
hypothesis = 'a man sitting on a blue bench.'

predict_inference(premise, hypothesis, TEXT, LABEL, model, device)

premise = 'a man sitting on a green bench.'
hypothesis = 'a male sat on a lime bench.'

predict_inference(premise, hypothesis, TEXT, LABEL, model, device)

premise = 'a man sitting on a green bench.'
hypothesis = 'a person on a park bench'

predict_inference(premise, hypothesis, TEXT, LABEL, model, device)

