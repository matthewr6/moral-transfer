import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BartModel, BartConfig

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, n_vocab=50264):
        self.n_vocab = n_vocab
        self.true_embedding = BartModel.from_pretrained('facebook/bart-large-cnn').encoder.embed_tokens

    def __len__(self):
        return self.n_vocab

    def __getitem__(self, index):
        ids = torch.LongTensor([index])
        one_hot = F.one_hot(ids, num_classes=self.n_vocab).float().detach()
        true_embeddings = self.true_embedding(ids).detach()
        return {
            'one_hot': one_hot,
            'true_embeddings': true_embeddings
        }


class PseudoEmbedding(pl.LightningModule):
    def __init__(self, args):
        super(PseudoEmbedding, self).__init__()
        self.hparams = args
        self.vocab_size = 50264
        self.embedding_size = 1024

        self.pseudo_embedding = nn.Linear(self.vocab_size, self.embedding_size, bias=False)
        self.pseudo_embedding.weight = nn.Parameter(self.build_lookups())

    def build_lookups(self):
        embeddings = BartModel.from_pretrained('facebook/bart-large-cnn').encoder.embed_tokens
        ids = torch.LongTensor([i for i in range(self.vocab_size)])
        return torch.transpose(embeddings(ids), 0, 1).detach()

    def forward(self, one_hot):
        print(one_hot.shape, self.pseudo_embedding.weight.shape)
        embeddings = self.pseudo_embedding(one_hot)
        return embeddings

    def loss_fn(self, outputs, targets):
        return torch.nn.MSELoss()(outputs, targets)

    def training_step(self, batch, batch_nb):
        one_hots = batch['one_hot']
        true_embeddings = batch['true_embeddings']
        pseudo_embeddings = self.forward(one_hots)
        loss = self.loss_fn(pseudo_embeddings, true_embeddings)
        self.log('train_loss', loss)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == '__main__':
    embedding_params = build_lookups()
