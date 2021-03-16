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
print("Start")
file = open('data/nela-covid-2020/combined/headlines_cnn_bart_split.pkl', 'rb')
data = pickle.load(file)
file.close()
print("Data Loaded")

test_dataset = NewsDataset(data['test'])


discriminator = OneHotMoralClassifier({}, use_mask=False)
discriminator.load_state_dict(torch.load('discriminator_titlemorals_state.pkl'))

model = MoralTransformer(discriminator=discriminator)
model.load_state_dict(torch.load('experiments/exp1/checkpoints/epoch=6-step=69999.ckpt')['state_dict'])
model = model.cuda()
model.eval()

vocab_size = 50264
# last_used_token = 50141
unused_tokens = [i for i in range(50264 - 10, 50264)]

def transfer(original_ids, original_mask, target_morals):
    ids_with_moral_tokens = original_ids[:]
    encdec_mask = original_mask[:]

    seq_end_idx = original_ids.index(2) + 1

    # extend sequences
    original_ids += [1] * 10
    ids_with_moral_tokens += [1] * 10
    original_mask += [0] * 10
    encdec_mask += [0] * 10

    # add morals and extend mask
    for i in range(10):
        if target_morals[i] == 1:
            ids_with_moral_tokens[seq_end_idx + i] = unused_tokens[i]
        encdec_mask[seq_end_idx + i] = 1

    original_ids = torch.LongTensor([original_ids]).cuda()
    ids_with_moral_tokens = torch.LongTensor([ids_with_moral_tokens]).cuda()
    original_mask = torch.LongTensor([original_mask]).cuda()
    encdec_mask = torch.LongTensor([encdec_mask]).cuda()
    target_morals = torch.FloatTensor([target_morals]).cuda()

    return np.array(model.forward(original_ids, ids_with_moral_tokens, original_mask, encdec_mask, target_morals).tolist())

for article_idx in [3245, 1544, 123, 432, 549, 1032]:
    article = test_dataset[article_idx]
    new_morals = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    print(article['target_morals'].tolist())
    print(new_morals)
    orig = np.array([article['original_ids'].tolist()])
    gen = transfer(article['original_ids'].tolist(), article['original_mask'].tolist(), new_morals)

    print(orig.shape, gen.shape)

    def convert(probs=None, tokens=None):
        if probs is not None:
            tokens = np.argmax(probs, 2)
        results = []
        for token_set in tokens:
            converted = model.tokenizer.convert_ids_to_tokens(token_set)
            sentence = ''.join(converted).replace('Ġ', ' ')
            results.append(sentence)
        return results

    print(convert(tokens=orig))
    print(convert(probs=gen))
