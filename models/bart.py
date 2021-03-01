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
    def __init__(self):
        super().__init__()
        self.Bart = BartModel.from_pretrained('facebook/bart-large-cnn') # https://huggingface.co/transformers/model_doc/bart.html#bartmodel, last_hidden_state 
        self.Bart = self.Bart.to(device)

    def forward(self, x):
        # x = (source, generated target, moral vec)
        # in lightning, forward defines the prediction/inference actions
        # embedding = self.encoder(x[0])
        # pred = self.decoder([embedding, x[1]])
        y = self.Bart(x)
        return y

    def training_step(self, batch, batch_idx):
        return
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # # Logging to TensorBoard by default
        # self.log('train_loss', loss)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



if __name__ == "__main__":
    trainer = pl.Trainer(
            gpus=0)

    model = MoralTransformer()
    trainer.fit(model)
    trainer.test()