import glob
import json
import os

data = []
for name in glob.glob('nela-covid-2020/preprocessed/*'):
    with open(name, 'r') as f:
        data += json.load(f)

with open('nela-covid-2020/combined.json', 'w') as f:
    json.dump(data, f)
