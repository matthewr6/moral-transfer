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
from models import MoralTransformer, MoralTransformerSpecial
from models.custom_transformer_classifier import OneHotMoralClassifier
from data import NewsDataset

import sys
# assert len(sys.argv) > 1
# exp_idx = int(sys.argv[1])
exp_idx = 0


# [moral tokens to, lr, moral mode, content loss metric, include moral loss]
experiments = [
    # ['decoder', 1e-6, 'identity', 'normalized_pairwise', False],
    # ['encoder', 1e-6, 'identity', 'normalized_pairwise', False],
    # ['decoder', 1e-6, 'random', 'normalized_pairwise', True],
    ['decoder', 1e-6, 'id+random', 'normalized_pairwise', True],
    # ['encoder', 1e-6, 'random', 'normalized_pairwise', True],

    # ['injection', 1e-6, 'identity', 'normalized_pairwise', False], # TODO IF TIME
    # ['injection', 1e-6, 'random', 'normalized_pairwise', True], # TODO IF TIME
]

def train(gpus):
    print("Loading data...")
    file = open('headlines_cnn_bart_split.pkl', 'rb')
    # file = open('data/nela-covid-2020/combined/headlines_cnn_bart_split.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    print("Data loaded")

    exp = experiments[exp_idx]

    feed_moral_tokens_to = exp[0]
    lr = exp[1]
    moral_mode = exp[2]
    use_content_loss = bool(exp[3])
    content_loss_type = exp[3]
    use_moral_loss = exp[4]

    exp_name = '_'.join([feed_moral_tokens_to, str(lr), moral_mode, str(content_loss_type), str(use_moral_loss)])
    exp_name = "resume" + exp_name 
    print(exp_name)

    # stuff to keep
    freeze_encoder = True
    freeze_decoder = False
    include_moral_tokens = True

    if feed_moral_tokens_to == 'injection':
        freeze_encoder = False
        include_moral_tokens = False

    data['train'] =  data['train']
    train_dataset = NewsDataset(data['train'], moral_mode=moral_mode, include_moral_tokens=include_moral_tokens)
    val_dataset = NewsDataset(data['val'], moral_mode=moral_mode, include_moral_tokens=include_moral_tokens)
    test_dataset = NewsDataset(data['test'], moral_mode=moral_mode, include_moral_tokens=include_moral_tokens)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)


    # ------------
    # training
    # ------------
    print('Loading discriminator...')
    discriminator = OneHotMoralClassifier({}, use_mask=False)
    discriminator.load_state_dict(torch.load('discriminator_titlemorals_state.pkl'))
    print('Discriminator loaded')

    model = MoralTransformerSpecial(
        lr=lr,
        discriminator=discriminator,
        use_content_loss=use_content_loss,
        contextual_injection=(not include_moral_tokens),
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
        feed_moral_tokens_to=feed_moral_tokens_to,
        content_loss_type=content_loss_type,
        use_moral_loss=use_moral_loss
    )

    # model.load_state_dict(torch.load('experiments/decoder_1e-06_id+random_normalized_pairwise_False/checkpoints/epoch=9-step=26589.ckpt')['state_dict'])


    checkpoint_callback= ModelCheckpoint(dirpath=os.path.join("./experiments", exp_name, "checkpoints"), save_top_k=1, save_last=True, monitor='train_loss', mode='min')
    trainer = Trainer(gpus=gpus, 
                    # auto_lr_find=False, # use to explore LRs
                    # distributed_backend='dp',
                    resume_from_checkpoint='experiments/decoder_1e-06_id+random_normalized_pairwise_False/checkpoints/epoch=9-step=26589.ckpt',
                    max_epochs=20,
                    callbacks=[checkpoint_callback],
                    )

    # LR Exploration        
    # lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
    # new_lr = lr_finder.suggestion()
    # print(new_lr)

    trainer.fit(model, train_loader, val_loader)

    with open(os.path.join("./experiments", exp_name, 'loss_history.pkl'), 'wb') as f:
        pickle.dump(model.loss_history, f)
    print(model.loss_history)

    print("Training Done")

# ------------
# testing
# ------------

if __name__ == '__main__':
    gpus = 1 if torch.cuda.is_available() else None
    train(gpus)