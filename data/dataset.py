
import pickle
import os
import datasets
from torch import nn
import torch
from tqdm import tqdm
from operator import itemgetter
import torch.nn.functional as F

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        labels = list(map(itemgetter('moral_features'), data))
        self.targets = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        article = self.data[index]
        ids = article['content']
        mask = article['attention_mask']
        # token_type_ids = article["token_type_ids"]
        targets = self.targets[index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.float)
        }