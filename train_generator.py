import pickle
import os
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import metrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models import MoralTransformer
from models.custom_transformer_classifier import OneHotMoralClassifier
from data import NewsDataset

# device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

def train(exp_name, gpus):
    print("Loading data...")
    file = open('data/nela-covid-2020/combined/headlines_cnn_bart_split.pkl', 'rb')
    # file = open('data/nela-covid-2020/combined/headlines_contentmorals_cnn_bart_split.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    print("Data loaded")

    # create datasets
    train_dataset = NewsDataset(data['train'])
    val_dataset = NewsDataset(data['val'])
    test_dataset = NewsDataset(data['test'])

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)


    # ------------
    # training
    # ------------
    LEARNING_RATE = 1e-5
    hparams = {'lr': LEARNING_RATE}
    print('Loading discriminator...')
    discriminator = OneHotMoralClassifier({}, use_mask=False)
    discriminator.load_state_dict(torch.load('epoch=5-step=7499.ckpt')['state_dict'])
    print('Discriminator loaded')
    model = MoralTransformer(discriminator=discriminator)
    # early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=True, mode='auto')
    # checkpoint_callback= ModelCheckpoint(dirpath=os.path.join("./experiments", exp_name, "checkpoints"), save_top_k=1, monitor='train_loss', mode='min')
    trainer = Trainer(gpus=gpus, 
                    # auto_lr_find=False, # use to explore LRs
                    # distributed_backend='dp',
                    max_epochs=20, 
                    # callbacks=[early_stop_callback, checkpoint_callback],
                    )

    # LR Exploration        
    lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
    print(lr_finder.results)
    fig = lr_finder.plot(suggest=True)
    # fig.show()
    # fig.savefig('lr.png')
    new_lr = lr_finder.suggestion()
    print(new_lr)

    trainer.fit(model, train_loader, val_loader)
    print("Training Done")

# ------------
# testing
# ------------

if __name__ == '__main__':
    gpus = 1 if torch.cuda.is_available() else None
    exp_name = 'GEN_TEST'
    train(exp_name, gpus)



