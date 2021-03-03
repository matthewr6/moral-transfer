import datasets
from datasets import load_dataset, load_metric, load_from_disk, Dataset
import pickle
import os
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from operator import itemgetter
import torch.nn.utils.rnn as rnn

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BartModel, BartConfig, DistilBertModel

from transformers import BartForSequenceClassification, BartTokenizer
from collections import Counter



class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.num_unique = self.get_num_unique()

        labels = list(map(itemgetter('moral_features'), data))
        max_vals = [max(idx) for idx in zip(*labels)] 
        normalized_labels = [ [ val/max_vals[index] if max_vals[index] > 0 else val for index,val in enumerate(row)] for row in labels] # moral feature wise normalization
        self.targets = [ [1 if i>= 0.5 else 0  for i in row] for row in normalized_labels]


    def __len__(self):
        return len(self.data)

    def get_num_unique(self):
        ids = np.array([a['content'][0] for a in self.data]).flatten()
        return ids.max()

    def __getitem__(self, index):
        article = self.data[index]
        ids = article['content'][0]
        # mask = article['attention_mask']
        # token_type_ids = article["token_type_ids"]
        targets = self.targets[index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            # 'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.float)
        }

print("Start")
# file = open('cnn_bart_encodings.pkl', 'rb')
file = open('../data/nela-covid-2020/combined/headlines_manual.pkl', 'rb')
data = pickle.load(file)
data = [d for d in data if sum(d['moral_features'])]
file.close()
print("Data Loaded")

# dummy to test pipeline
# dataset = NewsDataset(data[1:100])
dataset = NewsDataset(data)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-05
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(train_dataset, **train_params)
testing_loader = DataLoader(test_dataset, **test_params)

print("Training Examples: " + str(train_size))
print(len(training_loader))


class MoralClassifier(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=64, embedding_dim=128):
        super(MoralClassifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 11)

    def forward(self, ids):
        x = self.embeddings(ids)
        output, (h_n, c_n) = self.lstm(x)
        return self.linear(h_n[-1])


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = MoralClassifier(dataset.num_unique)
model = model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _,data in tqdm(enumerate(training_loader), "Training"):
        ids = data['ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%len(training_loader)==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)

print("Training Done")

def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0), "Testing: "):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            # token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

# validate
outputs, targets = validation(epoch)
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")