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
# file = open('headlines_cnn_bart_split.pkl', 'rb')
data = pickle.load(file)
file.close()
print("Data Loaded")

test_dataset = NewsDataset(data['test'])


discriminator = OneHotMoralClassifier({}, use_mask=False)
discriminator.load_state_dict(torch.load('saved_models/discriminator_titlemorals_state.pkl'))

model = MoralTransformer(discriminator=discriminator, feed_moral_tokens_to='decoder', contextual_injection=False)
# model.load_state_dict(torch.load('experiments/exp1/checkpoints/epoch=6-step=69999.ckpt')['state_dict'])
# model.load_state_dict(torch.load('experiments/decoder_1e-06_id+random_normalized_pairwise_False/checkpoints/epoch=9-step=26589.ckpt')['state_dict'])
# model.load_state_dict(torch.load('experiments/decoder_1e-06_identity_normalized_pairwise_False/checkpoints/epoch=17-step=95723.ckpt')['state_dict'])
# model.load_state_dict(torch.load('experiments/encoder_1e-06_identity_normalized_pairwise_False/checkpoints/epoch=14-step=79769.ckpt')['state_dict'])
# model.load_state_dict(torch.load('experiments/decoder_1e-06_random_normalized_pairwise_True/checkpoints/epoch=22-step=122313.ckpt')['state_dict'])
# model.load_state_dict(torch.load('experiments/encoder_1e-06_random_normalized_pairwise_True/checkpoints/epoch=23-step=127631.ckpt')['state_dict'])
model.load_state_dict(torch.load('experiments/RESUME decoder_1e-06_id+random_embedding_normalized_pairwise_True_content_weighted_10x/checkpoints/last.ckpt')['state_dict'])


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

def convert(probs=None, tokens=None):
    if probs is not None:
        tokens = np.argmax(probs, 2)
    results = []
    for token_set in tokens:
        # converted = model.tokenizer.convert_ids_to_tokens(token_set)
        # sentence = ''.join(converted).replace('Ġ', ' ')
        sentence = model.tokenizer.decode(token_set)
        stop_idx = len(sentence) + 1
        if '</s>' in sentence:
            stop_idx = sentence.index('</s>')
        results.append(sentence[3:stop_idx])
    return results

print('{} test samples'.format(len(test_dataset)))

while True:
    article_idx = len(test_dataset)
    while article_idx >= len(test_dataset):
        article_idx = int(input('Sample index: '))

    article = test_dataset[article_idx]
    original_morals = [int(v) for v in article['target_morals'].tolist()]
    print('Original morals:', original_morals)
    orig = np.array([article['original_ids'].tolist()])
    print(convert(tokens=orig)[0])
    target_morals = []
    while len(target_morals) != 10:
        target_morals = input('Target morals: ')
        try:
            target_morals = [int(v) for v in target_morals.split(',')]
        except:
            target_morals = []
    gen = transfer(article['original_ids'].tolist(), article['original_mask'].tolist(), target_morals)


    print(convert(probs=gen)[0])
    print('')
