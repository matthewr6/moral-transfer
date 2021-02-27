import glob
import json
import os
import random

with open('nela-covid-2020/combined.json', 'r') as f:
    data = json.load(f)

def moral_features_generator(batch_size=1):
    while True:
        X = []
        y = []
        for i in range(batch_size):
            article = random.choice(data)
            X.append(article['content'])
            y.append(article['moral_features'])
        yield X, y

def cyclic_generator(batch_size=1):
    while True:
        X = []
        y = []
        for i in range(batch_size):
            article = random.choice(data)
            X.append(article['content'])
            y.append(article['content'])
        yield X, y
