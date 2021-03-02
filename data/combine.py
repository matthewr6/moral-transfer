import glob
import json
import os
import pickle
from tqdm import tqdm

with open('filtered_sources.txt', 'r') as f:
    filtered_sources = [l.strip() for l in f.readlines()]

data = []
for name in tqdm(glob.glob('nela-covid-2020/newsdata/*')):
    sname = os.path.basename(name).split('.')[0]
    if sname not in filtered_sources:
        continue
    with open(name, 'r') as f:
        data += json.load(f)

with open('nela-covid-2020/combined/unprocessed.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
