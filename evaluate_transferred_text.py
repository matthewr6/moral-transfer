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
from transformers import BartTokenizerFast
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')

nela = NELAFeatureExtractor()
extractor = nela.extract_moral
moral_foundations = ['AuthorityVice', 'AuthorityVirtue', 'FairnessVice', 'FairnessVirtue', 'HarmVice', 'HarmVirtue', 'IngroupVice', 'IngroupVirtue', 'PurityVice', 'PurityVirtue']

def calc_generated_nela_features(text):
    try:
        feature_vector, feature_names = extractor(text) 
    except Exception as e:
        print(e)
        return None
    feature_dict = dict(zip(feature_names, feature_vector))
    return [1 if feature_dict[feature_name] > 0 else 0 for feature_name in moral_foundations]
    
def generated_nela_features(data):
    with multiprocessing.Pool(32) as pool:
        preprocessed = list(tqdm(pool.imap(calc_generated_nela_features, data), total=len(data)))
    return preprocessed

def convert(tokens):
    sentence = tokenizer.decode(tokens)
    stop_idx = len(sentence) + 1
    if '</s>' in sentence:
        stop_idx = sentence.index('</s>')
    return sentence[3:stop_idx]

def trim(string):
    if string[:3] == '<s>':
        string = string[3:]
    if '</s>' in string:
        idx = string.index('</s>')
        string = string[:idx]
    return string

results = pickle.load(open('results.pkl', 'rb'))

input_text = [convert(tokens=t) for t in results['original_ids']]
gen_text = [convert(tokens=t) for t in results['generated_probs']]

nela_morals = generated_nela_features(gen_text)

accuracy = metrics.accuracy_score(nela_morals, results['target_morals'])
f1_score_micro = metrics.f1_score(nela_morals, results['target_morals'], average='micro')
f1_score_macro = metrics.f1_score(nela_morals, results['target_morals'], average='macro')

print('Acc:', accuracy)
print('F1 micro:', f1_score_micro)
print('F1 macro:', f1_score_macro)

results['input_text'] = input_text
results['gen_text'] = gen_text
results['gen_nela_morals'] = nela_morals

pickle.dump(results, open('results_processed.pkl', 'wb'))

def get_target_moral_names(targets):
    r = []
    for idx, t in enumerate(targets):
        if t:
            r.append(moral_foundations[idx])
    return r

print(len(results['input_text']))
while True:
    try:
        idx = int(input('Sample idx: '))
        assert idx < len(results['input_text'])
    except:
        continue
    if idx < 0:
        break
    print('Input:  {}'.format(results['input_text'][idx]))
    print('Original morals: {}'.format(', '.join(get_target_moral_names(results['original_morals'][idx]))))
    print('Target morals:   {}'.format(', '.join(get_target_moral_names(results['target_morals'][idx]))))
    print('Output: {}'.format(results['gen_text'][idx]))
    print('NELA morals:     {}'.format(', '.join(get_target_moral_names(results['gen_nela_morals'][idx]))))
    print('')

