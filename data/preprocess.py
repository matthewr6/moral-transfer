import glob
import pickle
import json
import os
from tqdm import tqdm
import multiprocessing

from nela_features.nela_features import NELAFeatureExtractor, MORAL_FOUNDATION_DICT
from transformers import BartTokenizerFast, BertTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

moral_foundations = sorted(MORAL_FOUNDATION_DICT.keys())
print(moral_foundations, '\n')

nela = NELAFeatureExtractor()
extractor = nela.extract_moral

def add_moral_features(data):
    preprocessed = []
    for article in tqdm(data):
        try:
            feature_vector, feature_names = extractor(article['content']) 
        except:
            continue
        feature_dict = dict(zip(feature_names, feature_vector))
        article['moral_features'] = [feature_dict[feature_name] for feature_name in moral_foundations]
        preprocessed.append(article)
    return preprocessed

def add_headline_moral_features(data):
    preprocessed = []
    for article in tqdm(data):
        try:
            feature_vector, feature_names = extractor(article['title']) 
        except:
            continue
        feature_dict = dict(zip(feature_names, feature_vector))
        moral_features = [feature_dict[feature_name] for feature_name in moral_foundations]
        preprocessed.append({
            'content': article['title'],
            'moral_features': moral_features
        })
    return preprocessed

def strip_irrelevant_content(data, keys=['content', 'title', 'moral_features']):
    preprocessed = []
    for article in tqdm(data):
        a = {k: article[k] for k in keys}
        preprocessed.append(a)
    return preprocessed

def byte_pair_encoding(data):
    texts = [a['content'] for a in data]
    if os.path.exists('nela-covid-2020/tokenizers/bpe.json'):
        tokenizer = Tokenizer.from_file('nela-covid-2020/tokenizers/bpe.json')
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=50000, show_progress=True)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(texts, trainer)
        tokenizer.save('nela-covid-2020/tokenizers/bpe.json')
    encodings = tokenizer.encode_batch(texts)
    for idx, article in enumerate(tqdm(data)):
        # article['content'] = tokenizer.encode(article['content']).ids
        article['content'] = encodings[idx].ids
    return data

def bart_encoding(data):
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    texts = [a['content'] for a in data]
    encodings = tokenizer(texts, truncation=True, max_length=1024, padding=True, return_attention_mask=True, return_token_type_ids=True)
    for idx, article in enumerate(tqdm(data)):
        article['content'] = encodings.data['input_ids'][idx]
        article['attention_mask'] = encodings.data['attention_mask'][idx]
        article['token_type_ids'] = encodings.data['token_type_ids'][idx]
    return data

def cnn_bart_encoding(data):
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
    texts = [a['content'] for a in data]
    # encodings = tokenizer(texts, truncation=True, max_length=1024, padding=True, return_attention_mask=True, return_token_type_ids=True)
    encodings = tokenizer(texts, truncation=True, max_lenth=128, padding=True, return_attention_mask=True, return_token_type_ids=True)
    for idx, article in enumerate(tqdm(data)):
        article['content'] = encodings.data['input_ids'][idx]
        article['attention_mask'] = encodings.data['attention_mask'][idx]
        article['token_type_ids'] = encodings.data['token_type_ids'][idx]
    return data

def short_cnn_bart_encoding(data):
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
    texts = [a['content'] for a in data]
    encodings = tokenizer(texts, truncation=True, max_lenth=128, padding=True, return_attention_mask=True, return_token_type_ids=True)
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

def apply_preprocessing(infile, method, outfile):
    print(method.__name__)
    if os.path.exists(outfile):
        print('{} exists; skipping\n'.format(outfile))
        return
    if not os.path.exists(infile):
        print('{} does not exist; skipping\n'.format(infile))
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

# source = 'nela-elections-2020/combined' # headline max len: 109
source = 'nela-covid-2020/combined' 

apply_preprocessing('{}/unprocessed.pkl'.format(source), add_headline_moral_features, '{}/headlines.pkl'.format(source))
apply_preprocessing('{}/headlines.pkl'.format(source), short_cnn_bart_encoding, '{}/headlines_cnn_bart.pkl'.format(source))