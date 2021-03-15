
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
        if np.sum(output_vec) > 3:
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
        self.max_seq_len = 86 # can write a func to calculate this later
        self.num_morals = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        article = self.data[index]
        original_ids = article['content']
        mask = article['attention_mask']
        original_morals = self.targets[index]
        new_morals = original_morals[:]

        if self.moral_mode != 'identity':
            new_morals = rand_target_morals(original_morals)

        original_ids = ids[:]
        if self.include_moral_tokens:
            seq_end_idx = original_ids.index(STOP_TOKEN) + 1

            original_ids += [1] * self.num_morals
            original_ids += [1] * self.num_morals
            mask += [MASK] * self.num_morals

            for i in range(10):
                if original_morals[i] == 1:
                    ids[seq_end_idx + i] = unused_tokens[i]
                mask[seq_end_idx + i] = UNMASK

        return {
            'encoder_ids': torch.tensor(encoder_ids, dtype=torch.long),
            'decoder_ids': torch.tensor(decoder_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'original_ids': torch.tensor(original_ids, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(original_morals, dtype=torch.float)
        }

if __name__ == '__main__':
    a = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    output = rand_target_morals(a)
    print('test')
    print(output)
