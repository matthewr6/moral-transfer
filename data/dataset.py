
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
        if output_vec.sum() > 3:
            continue

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

# def rand_target_morals(input_vec):
#     unused_pair_idxs = []
#     for i in range(0, 10, 2):
#         if not input_vec[i] and not input_vec[i + 1]:
#             unused_pair_idxs.append(i)

#     output_vec = [0] * len(input_vec)
#     num_targets = np.random.randint(1, 3)
#     chosen_pair_idxs = np.random.choice(unused_pair_idxs, size=num_targets, replace=False)

#     for pair_idx in chosen_pair_idxs:
#         which = np.random.randint(0, 2)
#         output_vec[pair_idx + which] = 1

#     return output_vec

vocab_size = 50264
# last_used_token = 50141
unused_tokens = [i for i in range(50264 - 10, 50264)]

def calc_num_moral_types(arr):
    r = []
    for i in range(0, 10, 2):
        if arr[i] or arr[i + 1]:
            r.append(1)
        else:
            r.append(0)
    return sum(r)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, moral_mode='identity', include_moral_tokens=False):
        self.data = self.filter(data)
        self.moral_mode = moral_mode
        labels = list(map(itemgetter('moral_features'), data))
        self.targets = labels
        self.include_moral_tokens = include_moral_tokens
        self.max_seq_len = 86 # can write a func to calculate this later
        self.num_morals = 10

    def filter(self, data):
        filtered = []
        for d in data:
            morals = d['moral_features']
            num_moral_types = calc_num_moral_types(morals)
            if num_moral_types < 4:
                filtered.append(d)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        article = self.data[index]

        original_ids = article['content']
        ids_with_moral_tokens = original_ids[:]

        original_mask = article['attention_mask']
        encdec_mask = original_mask[:]

        original_morals = self.targets[index]
        target_morals = original_morals[:]

        if self.moral_mode != 'identity':
            target_morals = rand_target_morals(original_morals)

        if self.include_moral_tokens:
            seq_end_idx = original_ids.index(STOP_TOKEN) + 1

            # extend sequences
            original_ids += [1] * self.num_morals
            ids_with_moral_tokens += [1] * self.num_morals
            original_mask += [MASK] * self.num_morals
            encdec_mask += [MASK] * self.num_morals

            # add morals and extend mask
            for i in range(10):
                if target_morals[i] == 1:
                    ids_with_moral_tokens[seq_end_idx + i] = unused_tokens[i]
                encdec_mask[seq_end_idx + i] = UNMASK

        return {
            'original_ids':  torch.tensor(original_ids, dtype=torch.long),
            'ids_with_moral_tokens':   torch.tensor(ids_with_moral_tokens, dtype=torch.long),
            'original_mask': torch.tensor(original_mask, dtype=torch.long),
            'encdec_mask':   torch.tensor(encdec_mask, dtype=torch.long),
            'original_morals': torch.tensor(original_morals, dtype=torch.float)
            'target_morals': torch.tensor(target_morals, dtype=torch.float)
        }
