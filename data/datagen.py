import glob
import json
import os
import numpy as np
import random
import pickle

with open('nela-covid-2020/combined/headlines_cnn_bart_split.pkl', 'rb') as f:
    data = pickle.load(f)

UNMASK = 1
MASK = 0

START_TOK = 0
END_TOK = 2
PADDING_TOK = 1

# Generators give single epoch of data.

# Training data for discriminator model.
# Output format:
# X: [source sequence] (no source mask)
# y: [1, ...moral features]
def partial_sequence_generator(batch_size=1):
    training_data = data['train'].copy()
    random.shuffle(training_data)
    i = 0
    N = len(training_data)
    while i < N:
        X = []
        y = []
        next_batch_size = min(batch_size, N - i)
        for j in range(next_batch_size):
            article = training_data[i]

            # Get data
            input_seq = np.array(article['content'])
            input_mask = np.array(article['attention_mask'])
            target_morals = article['moral_features']

            # Generate partial seq
            seq_len = input_mask.sum()
            sample_seq_len = random.randint(0, seq_len - 1)
            generated_mask = np.zeros(input_mask.shape)
            generated_mask[:sample_seq_len] = UNMASK
            generated_mask[sample_seq_len:] = MASK

            X.append([
                input_seq.tolist(),
                # generated_mask.tolist()
            ])
            # y.append([1] + target_morals)
            y.append(target_morals)
            i += 1
        yield X, y


# Training data for generator model in one direction.
# Output format:
# X: [source sequence, original morals, empty sequence (as generated seq)]
# y: [source sequence, [1] + original morals]
def identity_moral_features_generator(batch_size=1):
    training_data = data['train'].copy()
    random.shuffle(training_data)
    i = 0
    N = len(training_data)
    while i < N:
        X = []
        y = []
        next_batch_size = min(batch_size, N - i)
        for j in range(next_batch_size):
            article = training_data[i]
            input_seq = article['content']
            input_mask = article['attention_mask']
            original_morals = article['moral_features']
            generated_seq = [PADDING_TOK] * len(input_seq)

            X.append([
                input_seq,
                original_morals,
                generated_seq
            ])
            y.append([
                input_seq,
                original_morals,
                # [1] + original_morals, # include discriminator prediction
            ])
            i += 1
        yield X, y

def gen_new_morals(moral_features):
    return moral_features

# Training data for generator model.
# Output format:
# X_initial: [source sequence, target morals, empty seq (as generated seq)]
# X_cyclic: [original morals, empty seq (as generated seq)]
# y: [source sequence, [1] + orignal morals]
# The process is auto-regressive in the sense that previously generated tokens are fed into the decoder to generate the next token
# in forward direction, pass in input sequence and target morals, autoregressively get output sequence
# in backward direction, use output sequence + oiginal morals, autoregressively get "orginal" seq
# use "original" seq for classifier
def cyclic_moral_features_generator(batch_size=1):
    training_data = data['train'].copy()
    random.shuffle(training_data)
    i = 0
    N = len(training_data)
    while i < N:
        input_seqs = []
        new_morals = []
        original_morals = []
        y = []
        next_batch_size = min(batch_size, N - i)
        for j in range(next_batch_size):
            article = training_data[i]

            # Get data
            input_seq = article['content']
            input_mask = article['attention_mask']
            original_moral = article['moral_features']
            new_moral = gen_new_morals(original_morals)

            input_seqs.append(input_seq)
            new_morals.append(new_moral)
            original_morals.append(original_moral)
            y.append(original_morals)
            # y.append([1] + original_morals)
            i += 1
        yield X_initial, X_cyclic, y

if __name__ == '__main__':
    generator = moral_features_generator(batch_size=100)
    i = 0
    for X, y in generator:
        i += len(X)
        # print(len(batch))
    print(i, len(data['train']))