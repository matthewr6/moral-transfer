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

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

class CustomMoralClassifier(pl.LightningModule):
    def __init__(self, args):
        super(MoralClassifier, self).__init__()
        self.hparams = args
        self.pseudoembedding = nn.Linear(vocab_size, embedding_dim) # TODO: what are these?

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=16, dim_feedforward=1024, dropout=0.1, activation='relu')
        self.encoder = nn.TransformerEncoder(decoder_layer, num_layers=12)
        # Pooler
        self.l2 = nn.Linear(1024, 1024)
        self.act = nn.Tanh()
        # Classifier
        self.l3 = nn.Dropout(0.2)
        self.l4 = nn.Linear(1024, 10) # 10 categories
    
    def loss_fn(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    # def forward(self, ids, mask):
    def forward(self, one_hot_encodings):
        pseudoembeddings = self.pseudoembedding(one_hot_encodings)
        encoded = eslf.encoder(pseudoembeddings).last_hidden_state
        output_2 = self.act(self.l2(encoded[:, 0]))
        output_3 = self.l3(output_2)
        output = self.l4(output_3)
        return output
    
    def training_step(self, batch, batch_nb):
        ids = batch['ids']
        mask = batch['mask']
        y = batch['targets']
        y_hat = self.forward(ids, mask)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        ids = batch['ids']
        mask = batch['mask']
        y = batch['targets']
        y_hat = self.forward(ids, mask)
        loss = self.loss_fn(y_hat, y)
        y_preds = (y_hat >= 0).int()  
        stats =  {'val_loss': loss, 
                   'progress_bar': {'val_loss': loss},
                   'y_preds': y_preds,
                   'y_hat': y_hat,
                   'y': y}
        
        self.log('val_loss', loss)
        return {**stats}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_preds = torch.cat([x['y_preds'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        accuracy = metrics.accuracy_score(y.cpu(), y_preds.cpu())
        f1_score_micro = metrics.f1_score(y.cpu(), y_preds.cpu(), average='micro')
        f1_score_macro = metrics.f1_score(y.cpu(), y_preds.cpu(), average='macro')

        stats = {
            'acc': accuracy,
            'f1-micro': f1_score_micro,
            'f1-macro': f1_score_macro
            }
        
        self.log('val_loss', avg_loss)
        print(stats)
        return {**stats}


    def test_step(self, batch, batch_nb):
        ids = batch['ids']
        mask = batch['mask']
        y = batch['targets']
        y_hat = self.forward(ids, mask)
        loss = self.loss_fn(y_hat, y)
        y_preds = (y_hat >= 0).int()  
        stats =  {'test_loss': loss, 
                   'progress_bar': {'test_loss': loss},
                   'y_preds': y_preds,
                   'y_hat': y_hat,
                   'y': y}
        
        self.log('test_loss', loss)
        return {**stats}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_preds = torch.cat([x['y_preds'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        accuracy = metrics.accuracy_score(y.cpu(), y_preds.cpu())
        f1_score_micro = metrics.f1_score(y.cpu(), y_preds.cpu(), average='micro')
        f1_score_macro = metrics.f1_score(y.cpu(), y_preds.cpu(), average='macro')

        stats = {
            'acc': accuracy,
            'f1-micro': f1_score_micro,
            'f1-macro': f1_score_macro
            }

        self.log('test_loss', avg_loss)
        print(stats)
        return {**stats}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    

