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

dataset = NewsDataset(data['test'], moral_mode='random')
dataloader = DataLoader(dataset, batch_size=8, num_workers=4)


discriminator = OneHotMoralClassifier({}, use_mask=False)
print('Loading discriminator...')
discriminator.load_state_dict(torch.load('saved_models/discriminator_titlemorals_state.pkl'))
print('Discriminator loaded')

model = MoralTransformer(discriminator=discriminator)
print('Loading generator state...')
model.load_state_dict(torch.load('experiments/RESUME decoder_1e-06_id+random_embedding_normalized_pairwise_True/checkpoints/last.ckpt')['state_dict'])
print('Generator state loaded')
model = model.cuda()
model.eval()

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

# {
#     'original_ids':  torch.tensor(original_ids, dtype=torch.long),
#     'ids_with_moral_tokens':   torch.tensor(ids_with_moral_tokens, dtype=torch.long),
#     'original_mask': torch.tensor(original_mask, dtype=torch.long),
#     'encdec_mask':   torch.tensor(encdec_mask, dtype=torch.long),
#     'original_morals': torch.tensor(original_morals, dtype=torch.float),
#     'target_morals': torch.tensor(target_morals, dtype=torch.float)
# }

def trim(string):
    if string[:3] == '<s>':
        string = string[3:]
    if '</s>' in string:
        idx = string.index('</s>')
        string = string[:idx]
    return string

# results = []
# for batch in tqdm(dataset):
# # for batch in dataset:
#     original_ids = batch['original_ids'].unsqueeze(0).cuda()
#     ids_with_moral_tokens = batch['ids_with_moral_tokens'].unsqueeze(0).cuda()

#     original_mask = batch['original_mask'].unsqueeze(0).cuda()
#     encdec_mask = batch['encdec_mask'].unsqueeze(0).cuda()
#     original_morals = batch['original_morals'].unsqueeze(0).cuda()
#     target_morals = batch['target_morals'].unsqueeze(0).cuda()

#     gen_probs = model(original_ids, ids_with_moral_tokens, original_mask, encdec_mask, target_morals).squeeze(0)
#     gen_tokens = np.argmax(gen_probs.tolist(), 1)

#     orig_string = model.tokenizer.decode(original_ids.squeeze(0))
#     new_string = model.tokenizer.decode(gen_tokens)
#     results.append({
#         'orig_tokens': original_ids.squeeze(0).tolist(),
#         'gen_tokens': original_ids.squeeze(0).tolist(),

#         'gen_probs': gen_probs.tolist(),

#         'orig_string': trim(orig_string),
#         'new_string': trim(new_string),

#         'orig_morals': original_morals.squeeze(0).tolist(),
#         'target_morals': target_morals.squeeze(0).tolist()
#     })
#     # gen_tokens = 
#     # print('')

trainer = Trainer(gpus=1)

with torch.no_grad():
    model.eval()
    res = trainer.test(model, dataloader)

print(res)

# pickle.dump(results, open('results.pkl', 'wb'))
# print(results)
