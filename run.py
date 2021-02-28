import datasets
from datasets import load_dataset, load_metric, load_from_disk
import pickle


with open('cnn_bart_encodings.pkl', 'rb') as f:
    data = pickle.load(f)
dataset = Dataset.from_dict(my_dict)
reloaded_encoded_dataset = load_from_disk("/cnn_bart_encodings.pkl")
import pdb; pdb.set_trace()



