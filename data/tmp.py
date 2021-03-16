import pickle
import numpy as np

d = pickle.load(open('nela-covid-2020/combined/headlines_contentmorals_cnn_bart_split.pkl', 'rb'))['train']

morals = [f['moral_features'] for f in d]

def func(arr):
    r = []
    for i in range(0, 10, 2):
        if arr[i] or arr[i + 1]:
            r.append(1)
        else:
            r.append(0)
    return sum(r)

moral_condensed = np.array([func(d) for d in morals])
am = np.argmax(moral_condensed)
# print(am, morals[am])

# print((moral_condensed == 5).sum())
# print((moral_condensed == 4).sum())
# print((moral_condensed == 3).sum())
# print((moral_condensed == 2).sum())
# print((moral_condensed == 1).sum())
# print((moral_condensed == 0).sum())

# print(moral_condensed.mean())


def rand_target_morals(input_vec):
    unused_pair_idxs = []
    for i in range(0, 10, 2):
        if not input_vec[i] and not input_vec[i + 1]:
            unused_pair_idxs.append(i)

    output_vec = [0] * len(input_vec)
    num_targets = np.random.randint(1, 3)
    chosen_pair_idxs = np.random.choice(unused_pair_idxs, size=num_targets, replace=False)

    for pair_idx in chosen_pair_idxs:
        which = np.random.randint(0, 2)
        output_vec[pair_idx + which] = 1

    return output_vec

print(rand_target_morals([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]))