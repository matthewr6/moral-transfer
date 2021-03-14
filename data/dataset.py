
import pickle
import os
import datasets
from torch import nn
import torch
from tqdm import tqdm
from operator import itemgetter
import torch.nn.functional as F

import numpy as np


def rand_target_morals(input_vec):
    assert len(input_vec) == 10

    while True:
        output_vec = np.random.randint(0, 2, 10)  # randomly generate output moral

        # Check similarity. Output should be different from the input
        combined_vec = zip(input_vec, output_vec)
        different = False
        for moral in combined_vec:
            if moral[0] != moral[1]:
                different = True
                break

        if different:
            # Check for opposing morals (e.g. care vs harm) - both can't be 1
            morals_consistent = True
            for i in range(0, 10, 2):
                if output_vec[i] == output_vec[i+1] == 1:
                    morals_consistent = False
            if morals_consistent:
                return output_vec  # No opposing morals, return True


# print(rand_target_morals([0 for i in range(10)]))


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, moral_mode='identity'):
        self.data = data
        self.moral_mode = moral_mode
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

        if self.moral_mode != 'identity':
            targets = rand_target_morals(targets)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.float)
        }