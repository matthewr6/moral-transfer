
import pickle
import os
import datasets
from torch import nn
import torch
from tqdm import tqdm
from operator import itemgetter
import torch.nn.functional as F

import numpy as np

STOP_TOKEN = 2
MASK = 0
UNMASK = 1

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

vocab_size = 50264
# last_used_token = 50141
unused_tokens = [i for i in range(50264 - 10, 50264)]

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, moral_mode='identity', include_moral_tokens=False):
        self.data = data
        self.moral_mode = moral_mode
        labels = list(map(itemgetter('moral_features'), data))
        self.targets = labels
        self.include_moral_tokens = include_moral_tokens

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

        original_ids = ids[:]
        if self.include_moral_tokens:
            seq_end_idx = ids.index(STOP_TOKEN) + 1

            seq_len = len(ids)
            target_len = seq_end_idx + 10
            len_diff = target_len - seq_len
            if len_diff > 0:
                ids += [1] * len_diff
                original_ids += [1] * len_diff
                mask += [MASK] * len_diff

            for i in range(10):
                if targets[i] == 1:
                    ids[seq_end_idx + i] = unused_tokens[i]
                mask[seq_end_idx + i] = UNMASK

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'original_ids': torch.tensor(original_ids, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.float)
        }