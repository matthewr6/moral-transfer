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
END_TOK = 0
PADDING_TOK = 1

# Generators give single epoch of data.

# Output format:
# X: [source sequence, source mask, target morals, generated sequence, generated mask]
# y: [output sequence, dicriminato output]
def moral_features_generator(batch_size=1):
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

            # Partial seq with next token
            output_seq = input_seq.copy()
            output_seq[sample_seq_len + 1:] = PADDING_TOK

            X.append([
                input_seq.tolist(),
                input_mask.tolist(),
                target_morals,
                input_seq.tolist(), # can reuse due to attention masking!
                generated_mask.tolist()
            ])
            y.append([
                output_seq.tolist(),
                [1] + target_morals, # include discriminator prediction
            ])
            i += 1
        yield X, y

if __name__ == '__main__':
    generator = moral_features_generator(batch_size=100)
    i = 0
    for X, y in generator:
        i += len(X)
        # print(len(batch))
    print(i, len(data['train']))