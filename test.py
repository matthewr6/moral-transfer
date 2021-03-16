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
from models import MoralClassifier
from models.custom_transformer_classifier import OneHotMoralClassifier
from data import NewsDataset
import torch


def test(path, gpus):
    # load 
    print("Start")
    file = open('data/nela-covid-2020/combined/headlines_contentmorals_cnn_bart_split.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    print("Data Loaded")

    test_dataset = NewsDataset(data['test'])
    # test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4)

    # model = OneHotMoralClassifier.load_from_checkpoint(path)
    model = OneHotMoralClassifier({}, use_mask=False)
    model.load_state_dict(torch.load(path))
    trainer = Trainer(gpus=gpus, 
                      distributed_backend='dp')

    trainer.test(model, test_dataloaders=test_loader)
    
if __name__ == '__main__':
    # gpus = torch.cuda.device_count() if torch.cuda.is_available() else None
    gpus = 1 if torch.cuda.is_available() else None
    path = "dicriminator_contentmorals_state.pkl"
    test(path, gpus)
