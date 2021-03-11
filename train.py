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
from data import NewsDataset
import torch

# device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

print("Start")
file = open('headlines_cnn_bart_split.pkl', 'rb')
data = pickle.load(file)
file.close()
print("Data Loaded")

# create datasets
# train_dataset = NewsDataset(data['train'][0:1])
train_dataset = NewsDataset(data['train'])
val_dataset = NewsDataset(data['val'])
test_dataset = NewsDataset(data['test'])

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

# ------------
# training
# ------------
LEARNING_RATE = 1e-5
hparams = {'lr': LEARNING_RATE}
model = MoralClassifier(hparams)
# model = model.to(device)
early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=True, mode='auto')
checkpoint_callback= ModelCheckpoint(dirpath=os.path.join("./experiments", "test", "checkpoints"), save_top_k=1, monitor='val_loss', mode='min')
trainer = Trainer(gpus=1, 
                  distributed_backend='dp',
                  max_epochs=20, 
                  callbacks=[early_stop_callback, checkpoint_callback],
                  )
                     
trainer.fit(model, train_loader, val_loader)
print("Training Done")

# ------------
# testing
# ------------

