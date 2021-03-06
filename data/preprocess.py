import glob
import pickle
import json
import os
import re
import string
import collections
import numpy as np
from tqdm import tqdm
import multiprocessing

import spacy
from nela_features.nela_features import NELAFeatureExtractor, MORAL_FOUNDATION_DICT
from transformers import BartTokenizerFast, BertTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

moral_foundations = ['AuthorityVice', 'AuthorityVirtue', 'FairnessVice', 'FairnessVirtue', 'HarmVice', 'HarmVirtue', 'IngroupVice', 'IngroupVirtue', 'PurityVice', 'PurityVirtue']

# subversion authority cheating fairness harm care betrayal loyalty degradation purity

nela = NELAFeatureExtractor()
extractor = nela.extract_moral

def handle_content_moral_features(article):
    try:
        feature_vector, feature_names = extractor(article['content']) 
    except:
        return None
    feature_dict = dict(zip(feature_names, feature_vector))
    moral_features = [feature_dict[feature_name] for feature_name in moral_foundations]
    moral_features = [1 if v > 0 else 0 for v in moral_features]
    if sum(moral_features) == 0:
        return None
    return {
        'content': article['title'],
        'moral_features': moral_features
    }
def add_content_moral_features(data):
    with multiprocessing.Pool(32) as pool:
        preprocessed = list(tqdm(pool.imap_unordered(handle_content_moral_features, data), total=len(data)))
    return [a for a in preprocessed if a is not None]

def handle_headline_moral_features(article):
    try:
        feature_vector, feature_names = extractor(article['title']) 
    except:
        return None
    feature_dict = dict(zip(feature_names, feature_vector))
    moral_features = [feature_dict[feature_name] for feature_name in moral_foundations]
    moral_features = [1 if v > 0 else 0 for v in moral_features]
    if sum(moral_features) == 0:
        return None
    return {
        'content': article['title'],
        'moral_features': moral_features
    }
def add_headline_moral_features(data):
    with multiprocessing.Pool(32) as pool:
        preprocessed = list(tqdm(pool.imap_unordered(handle_headline_moral_features, data), total=len(data)))
    return [a for a in preprocessed if a is not None]

def cnn_bart_encoding(data):
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
    texts = [a['content'] for a in data]
    encodings = tokenizer(texts, truncation=True, max_length=128, padding=True, return_attention_mask=True, return_token_type_ids=True)
    for idx, article in enumerate(tqdm(data)):
        article['content'] = encodings.data['input_ids'][idx]
        article['attention_mask'] = encodings.data['attention_mask'][idx]
        article['token_type_ids'] = encodings.data['token_type_ids'][idx]
    return data

def short_cnn_bart_encoding(data):
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
    texts = [a['content'] for a in data]
    encodings = tokenizer(texts, truncation=True, max_length=128, padding=True, return_attention_mask=True, return_token_type_ids=True)
    for idx, article in enumerate(tqdm(data)):
        article['content'] = encodings.data['input_ids'][idx]
        article['attention_mask'] = encodings.data['attention_mask'][idx]
        article['token_type_ids'] = encodings.data['token_type_ids'][idx]
    return data

def distilbert_encoding(data):
    tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-uncased')
    texts = [a['content'] for a in data]
    encodings = tokenizer(texts, truncation=True, max_length=512, padding=True, return_attention_mask=True, return_token_type_ids=True)
    for idx, article in enumerate(tqdm(data)):
        article['content'] = encodings.data['input_ids'][idx]
        article['attention_mask'] = encodings.data['attention_mask'][idx]
        article['token_type_ids'] = encodings.data['token_type_ids'][idx]
    return data

# From https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
def manual_preprocessing(data):
    tok = spacy.load('en')
    def tokenize (text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        return [token.text for token in tok.tokenizer(nopunct)]
    counts = collections.Counter()
    for article in tqdm(data):
        counts.update(tokenize(article['content']))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    def encode_sentence(text, vocab2index, N=128):
        tokenized = tokenize(text)
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded.tolist(), length
    for article in tqdm(data):
        article['content'] = encode_sentence(article['content'], vocab2index)
    return data

def apply_preprocessing(infile, method, outfile):
    print(method.__name__)
    if not os.path.exists(infile):
        print('{} does not exist; skipping\n'.format(infile))
        return
    if os.path.exists(outfile):
        continue_anyways = input('{} exists; overwrite? (y/n) '.format(outfile)).lower()
        if 'y' not in continue_anyways:
            print('Skipping\n'.format(outfile))
            return
    print('Loading...')
    if 'json' in infile:
        with open(infile, 'r') as f:
            data = json.load(f)
    else:
        with open(infile, 'rb') as f:
            data = pickle.load(f)
    print('Preprocessing...')
    preprocessed = method(data)
    print('Saving...')
    with open(outfile, 'wb') as f:
        # json.dump(preprocessed, f)
        pickle.dump(preprocessed, f, pickle.HIGHEST_PROTOCOL)
    print('')

source = 'nela-elections-2020/combined'
# source = 'nela-covid-2020/combined'

preprocessing_steps = [
    {
        'in': 'unprocessed',
        'method': add_headline_moral_features,
        'out': 'headlines'
    },
    {
        'in': 'headlines',
        'method': short_cnn_bart_encoding,
        'out': 'headlines_cnn_bart'
    },
    {
        'in': 'headlines',
        'method': manual_preprocessing,
        'out': 'headlines_manual'
    },

    # {
    #     'in': 'unprocessed',
    #     'method': add_content_moral_features,
    #     'out': 'headlines_contentmorals'
    # },
    # {
    #     'in': 'headlines',
    #     'method': short_cnn_bart_encoding,
    #     'out': 'headlines_contentmorals_cnn_bart'
    # },
    # {
    #     'in': 'headlines',
    #     'method': manual_preprocessing,
    #     'out': 'headlines_contentmorals_manual'
    # },
]

for step in preprocessing_steps:
    apply_preprocessing(
        '{}/{}.pkl'.format(source, step['in']),
        step['method'],
        '{}/{}.pkl'.format(source, step['out']),
    )
