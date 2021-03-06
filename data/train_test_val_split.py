import os
import random
import glob
import pickle

train_percent = 80
test_percent = 10
val_percent = 10

def split_data(data):
    n = len(data)
    random.shuffle(data)
    train_count = int(n * train_percent / 100)
    test_count = int(n * test_percent / 100)
    return {
        'train': data[:train_count],
        'test': data[train_count:train_count + test_count],
        'val': data[train_count + test_count:]
    }

# datadir = 'nela-covid-2020/combined'
datadir = 'nela-elections-2020/combined'

for datafile in glob.glob('{}/*.pkl'.format(datadir)):
    if '_split.pkl' in datafile:
        continue
    basename = os.path.basename(datafile).split('.')[0]
    with open(datafile, 'rb') as f:
        data = pickle.load(f)
    split = split_data(data)
    with open(os.path.join(datadir, '{}_split.pkl'.format(basename)), 'wb') as f:
        pickle.dump(split, f)
