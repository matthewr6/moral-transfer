import pickle
import os
import torch
import sys
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models import MoralClassifier
from models.custom_transformer_classifier import OneHotMoralClassifier
from data import NewsDataset
import torch
from models import MoralTransformer


# load 
print("Loading data...")
file = open('data/nela-covid-2020/combined/headlines_cnn_bart_split.pkl', 'rb')
# file = open('headlines_cnn_bart_split.pkl', 'rb')
data = pickle.load(file)
file.close()
print("Data loaded")

dataset = NewsDataset(data['test'], moral_mode='random', include_moral_tokens=True)
dataloader = DataLoader(dataset, batch_size=64, num_workers=4)


discriminator = OneHotMoralClassifier({}, use_mask=False)
print('Loading discriminator...')
discriminator.load_state_dict(torch.load('final_models/discriminator_titlemorals_state.pkl'))
print('Discriminator loaded')

model = MoralTransformer(discriminator=discriminator, feed_moral_tokens_to='decoder', contextual_injection=False)
print('Loading generator state...')
model.load_state_dict(torch.load('final_models/special_finetuned.ckpt')['state_dict'])
# model.load_state_dict(torch.load('experiments/RESUME decoder_1e-06_id+random_embedding_normalized_pairwise_True/checkpoints/last.ckpt')['state_dict'])
print('Generator state loaded')

def convert(probs=None, tokens=None):
    if probs is not None:
        tokens = np.argmax(probs, 2)
    results = []
    for token_set in tokens:
        sentence = model.tokenizer.decode(token_set)
        stop_idx = len(sentence) + 1
        if '</s>' in sentence:
            stop_idx = sentence.index('</s>')
        results.append(sentence[3:stop_idx])
    return results

def trim(string):
    if string[:3] == '<s>':
        string = string[3:]
    if '</s>' in string:
        idx = string.index('</s>')
        string = string[:idx]
    return string

trainer = Trainer(gpus=1)
with torch.no_grad():
    model.eval()
    res = trainer.test(model, dataloader)[0]

pickle.dump(res, open('results.pkl', 'wb'))
# print(results)
