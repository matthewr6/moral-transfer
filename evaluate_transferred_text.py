import glob
import pickle
import json
import os
import torch
import torch.nn.functional as F
import re
import string
import collections
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import multiprocessing

import spacy
from nela_features.nela_features import NELAFeatureExtractor, MORAL_FOUNDATION_DICT
from models.custom_transformer_classifier import OneHotMoralClassifier

nela = NELAFeatureExtractor()
extractor = nela.extract_moral
moral_foundations = ['AuthorityVice', 'AuthorityVirtue', 'FairnessVice', 'FairnessVirtue', 'HarmVice', 'HarmVirtue', 'IngroupVice', 'IngroupVirtue', 'PurityVice', 'PurityVirtue']

print('Loading discriminator...')
discriminator = OneHotMoralClassifier({}, use_mask=False)
discriminator.load_state_dict(torch.load('saved_models/discriminator_titlemorals_state.pkl'))
print('Discriminator loaded')

def calc_generated_nela_features(article):
    try:
        feature_vector, feature_names = extractor(article['new_string']) 
    except Exception as e:
        print(e)
        return None
    feature_dict = dict(zip(feature_names, feature_vector))
    article['generated_nela_morals'] = [1 if feature_dict[feature_name] > 0 else 0 for feature_name in moral_foundations]
    return article
    
def add_generated_nela_features(data):
    with multiprocessing.Pool(32) as pool:
        preprocessed = list(tqdm(pool.imap(calc_generated_nela_features, data), total=len(data)))
    return preprocessed

def remove_post_end_tokens(tokens):
    if 2 not in tokens:
        return tokens
    end_idx = tokens.index(2)
    for i in range(end_idx + 1, len(tokens)):
        tokens[i] = 1
    return tokens

def add_discriminator_predictions(data):
    for article in tqdm(data):
        tokens = torch.LongTensor([article['gen_tokens'], remove_post_end_tokens(article['gen_tokens'])])
        one_hot_tokens = F.one_hot(tokens, num_classes=50264).float()
        res = discriminator.forward(one_hot_tokens).tolist()
        article['classifier_prediction'] = [1 if v >= 0 else 0 for v in res[0]]
        article['classifier_prediction_pruned'] = [1 if v >= 0 else 0 for v in res[1]]
    return data

# data = pickle.load(open('results.pkl', 'rb'))
# data = add_generated_nela_features(data)
# data = add_discriminator_predictions(data)
# pickle.dump(data, open('results2.pkl', 'wb'))

data = pickle.load(open('results2.pkl', 'rb'))

target_morals = [d['target_morals'] for d in data]

# truth = [d['generated_nela_morals'] for d in data]
# truth = [d['classifier_prediction'] for d in data]
truth = [d['classifier_prediction_pruned'] for d in data]

accuracy = metrics.accuracy_score(truth, target_morals)
f1_score_micro = metrics.f1_score(truth, target_morals, average='micro')
f1_score_macro = metrics.f1_score(truth, target_morals, average='macro')

print('Acc:', accuracy)
print('F1 micro:', f1_score_micro)
print('F1 macro:', f1_score_macro)
