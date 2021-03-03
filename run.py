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
from transformers import BertTokenizer, BertModel, BertConfig
from operator import itemgetter

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BartModel, BartConfig, DistilBertModel

from transformers import BartForSequenceClassification, BartTokenizer
from collections import Counter



class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

        labels = list(map(itemgetter('moral_features'), data))
        max_vals = [max(idx) for idx in zip(*labels)] 
        normalized_labels = [ [ val/max_vals[index] if max_vals[index] > 0 else val for index,val in enumerate(row)] for row in labels] # moral feature wise normalization
        self.targets = [ [1 if i>= 0.5 else 0  for i in row] for row in normalized_labels]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        article = self.data[index]
        ids = article['content']
        mask = article['attention_mask']
        # token_type_ids = article["token_type_ids"]
        targets = self.targets[index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.float)
        }

print("Start")
# file = open('cnn_bart_encodings.pkl', 'rb')
file = open('data/nela-covid-2020/combined/headlines_cnn_bart.pkl', 'rb')
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
    def __init__(self):
        super(MoralClassifier, self).__init__()
        # self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.l1 = BartModel.from_pretrained('facebook/bart-large-cnn')
        # Pooler
        self.l2 = torch.nn.Linear(1024, 1024)
        self.act = torch.nn.Tanh()
        # Classifier
        self.l3 = torch.nn.Dropout(0.3)
        self.l4 = torch.nn.Linear(1024, 11) # 11 categories

    def forward(self, ids, mask):
        output_1 = self.l1.encoder(ids, attention_mask = mask).last_hidden_state
        output_2 = self.act(self.l2(output_1[:, 0]))
        output_3 = self.l3(output_2)
        output = self.l4(output_3)
        return output


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = MoralClassifier()
model = model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0), "Training: "):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        # token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        # outputs = model(ids, mask, token_type_ids)
        outputs = model(ids, mask)

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
            targets = data['targets'].to(device, dtype = torch.int)
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