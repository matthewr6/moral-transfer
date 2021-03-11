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

# Inputs:
#   - input sequence (to encoder)
#   - currently generated text (to decoder)
#   - moral vector (to decoder)
# Outputs:
#   - next word
# https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
# context vector concat with moral vector as input to decoder transformer
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
# stack https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html after BERT
# https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83 --> include encoder outputs at every decoder level --> could include moral vector to
# https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

class MoralClassifier(pl.LightningModule):
    def __init__(self, args):
        super(MoralClassifier, self).__init__()
        self.hparams = args
        self.l1 = BartModel.from_pretrained('facebook/bart-large-cnn')
        # freeze bert weights
        for param in self.l1.parameters():
            param.requires_grad = False        
        # Pooler
        self.l2 = torch.nn.Linear(1024, 1024)
        self.act = torch.nn.Tanh()
        # Classifier
        self.l3 = torch.nn.Dropout(0.2)
        self.l4 = torch.nn.Linear(1024, 10) # 10 categories
    
    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def forward(self, ids, mask):
        output_1 = self.l1.encoder(ids, attention_mask = mask).last_hidden_state
        output_2 = self.act(self.l2(output_1[:, 0]))
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
        metrics =  {'val_loss': loss, 
                   'progress_bar': {'val_loss': loss},
                   'y_preds': y_preds,
                   'y_hat': y_hat,
                   'y': y}
        
        self.log('val_loss', loss)
        return {**metrics}
    
    def validation_end(self, outputs):
        import pdb; pdb.set_trace();
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_preds = torch.cat([x['y_preds'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        accuracy = metrics.accuracy_score(y, y_preds)
        f1_score_micro = metrics.f1_score(y, y_preds, average='micro')
        f1_score_macro = metrics.f1_score(y, y_preds, average='macro')
        
        metrics = {
            'acc': accuracy,
            'f1-micro': f1_score_micro,
            'f1-macro': f1_score_macro
            }
        
        self.log('val_loss', avg_loss, **metrics)
        print(metrics)
        return {**metrics}


    def test_step(self, batch, batch_nb):
        ids = batch['ids']
        mask = batch['mask']
        y = batch['targets']
        y_hat = self.forward(ids, mask)
        loss = self.loss_fn(y_hat, y)
        y_preds = (y_hat >= 0).int()  
        metrics =  {'test_loss': loss, 
                   'progress_bar': {'test_loss': loss},
                   'y_preds': y_preds,
                   'y_hat': y_hat,
                   'y': y}
        
        self.log('test_loss', loss)
        return {**metrics}
    
    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_preds = torch.cat([x['y_preds'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        accuracy = metrics.accuracy_score(y, y_preds)
        f1_score_micro = metrics.f1_score(y, y_preds, average='micro')
        f1_score_macro = metrics.f1_score(y, y_preds, average='macro')

        metrics = {
            'acc': accuracy,
            'f1-micro': f1_score_micro,
            'f1-macro': f1_score_macro
            }

        self.log('test_loss', avg_loss, **metrics)
        print(metrics)
        return {**metrics}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    

