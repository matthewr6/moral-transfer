import glob
import json
import os

with open('filtered_sources.txt', 'r') as f:
    filtered_sources = [l.strip() for l in f.readlines()]

data = []
for name in glob.glob('nela-covid-2020/newsdata/*'):
    sname = os.path.basename(name).split('.')[0]
    if sname not in filtered_sources:
        continue
    with open(name, 'r') as f:
        data += json.load(f)

with open('nela-covid-2020/combined/unprocessed.json', 'w') as f:
    json.dump(data, f)
