import logging

FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

from model import MoralClassifier, Embedding, LSTM, Linear
import torch
from torch import nn


# models = {}
# optimizers = {}
# if mode == 'train':
#     word_embedding = Embedding(len(token_vocab), embedding_dim, padding_idx=0, sparse=True, pretrain=embedding_file, vocab=token_vocab, trainable=True)
#     for target_label in labels:
#         lstm = LSTM(embedding_dim, hidden_size, batch_first=True, forget_bias=1.0)
#         linears = [Linear(i, o) for i, o in zip([hidden_size] + linear_sizes, linear_sizes + [2])]
#         model = MoralClassifier(word_embedding, lstm, linears)
#         if use_gpu:
#             model.cuda()
#         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=.9)
#         models[target_label] = model
#         optimizers[target_label] = optimizer


lstm = LSTM(1, 1)
word_embedding = nn.Embedding(1, 1)
linears = [nn.Linear(1, 1)]
model = MoralClassifier(word_embedding, lstm, linears)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(10):
    optimizer.zero_grad()
    x = torch.zeros(1, 1).long()
    lens = torch.tensor([1])
    out = model(x, lens)
    out.backward()
    optimizer.step()
