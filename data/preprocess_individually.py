import glob
import json
import os
from nela_features.nela_features import NELAFeatureExtractor

nela = NELAFeatureExtractor()
extractor = nela.extract_moral

def preprocess_content(content):
    return content

with open('filtered_covid_sources.txt', 'r') as f:
    filtered_sources = [l.strip() for l in f.readlines()]

for name in glob.glob('nela-covid-2020/newsdata/*'):
    sname = os.path.basename(name).split('.')[0]
    if sname not in filtered_sources or os.path.exists('nela-covid-2020/preprocessed/{}.json'.format(sname)):
        continue

    with open(name, 'r') as f:
        data = json.load(f)

    preprocessed = []
    for article in data:
        try:
            feature_vector, feature_names = extractor(article['content']) 
        except:
            continue
        article['moral_features'] = feature_vector
        preprocessed.append(article)

    with open('nela-covid-2020/preprocessed/{}.json'.format(sname), 'w') as f:
        json.dump(preprocessed, f)
