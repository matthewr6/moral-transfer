import pickle
import random
import csv
from operator import itemgetter
import numpy as np

with open('data/headlines.pkl', 'rb') as f:
    headlines = pickle.load(f)
    headlines = [line for line in headlines if sum(line['moral_features'])]  # remove vectors with only 0s

labels = list(map(itemgetter('moral_features'), headlines))
max_vals = [max(idx) for idx in zip(*labels)]
normalized_labels = [[val / max_vals[index] if max_vals[index] > 0 else val for index, val in enumerate(row)] for row in labels]  # moral feature wise normalization
targets = [np.array([1 if i > 0 else 0 for i in row]) for row in normalized_labels]

for line_idx in range(len(headlines)):
    label_vector = targets[line_idx]
    labels = ""
    for i in range(len(label_vector)):
        if label_vector[i] == 1:
            if i == 0:
                labels += "AB,"
            elif i == 1:
                labels += "AG,"
            elif i == 2:
                labels += "FB,"
            elif i == 3:
                labels += "FG,"
            elif i == 4:
                labels += "HB,"
            elif i == 5:
                labels += "HG,"
            elif i == 6:
                labels += "IB,"
            elif i == 7:
                labels += "IG,"
            elif i == 8:
                labels += "MG,"
            elif i == 9:
                labels += "PB,"
            elif i == 10:
                labels += "PG,"
    if labels != "":
        labels = labels[:-1]
    else:
        print("No label provided for this string.")

    headlines[line_idx]['moral_features'] = labels

random.seed(230)
random.shuffle(headlines)

with open('data/headlines_lstm.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for line in headlines:
        writer.writerow(['-', line['content'], line['moral_features']])

split_1 = int(0.8 * len(headlines))
split_2 = int(0.9 * len(headlines))
training_set = headlines[:split_1]
dev_set = headlines[split_1:split_2]
test_set = headlines[split_2:]

with open('data/training.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for line in training_set:
        writer.writerow(['-', line['content'], line['moral_features']])

with open('data/dev.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for line in dev_set:
        writer.writerow(['-', line['content'], line['moral_features']])

with open('data/test.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for line in test_set:
        writer.writerow(['-', line['content'], line['moral_features']])
