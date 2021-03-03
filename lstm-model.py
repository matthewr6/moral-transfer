import torch
from torch import nn
import torch.nn.init
import torch.nn.utils.rnn as rnn
import torch.nn.functional
# import train.py

class Model(nn.LSTM):
    def __init__(self, dataset):

        self.input_size = 128
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = nn.Dropout(0.3)
        self.embedding_dim = 128

        self.bias = True
        self.batch_first = False
        self.bidirectional = True

        super(Model, self).__init__(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bias=self.bias,
                                    batch_first=self.batch_first,
                                    dropout=self.dropout,
                                    bidirectional=self.bidirectional,
                                    )

        vocab_len = len(dataset.unique_words)
        self.embeddings = nn.Embedding(vocab_len, self.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, 5)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = rnn.pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        return self.linear(ht[-1])

# train(model, epochs=30, lr=0.05)
