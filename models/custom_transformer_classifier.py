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
from transformers import BartTokenizerFast


from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from collections import defaultdict

from transformers import BartModel, BartConfig

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')

moral_foundations = ['AuthorityVice', 'AuthorityVirtue', 'FairnessVice', 'FairnessVirtue', 'HarmVice', 'HarmVirtue', 'IngroupVice', 'IngroupVirtue', 'PurityVice', 'PurityVirtue']
def get_target_moral_names(targets):
    r = []
    for idx, t in enumerate(targets):
        if t:
            r.append(moral_foundations[idx])
    return r


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

class OneHotMoralClassifier(pl.LightningModule):
    def __init__(self, args, use_mask=True):
        super(OneHotMoralClassifier, self).__init__()
        self.hparams = args
        self.bart = BartModel.from_pretrained('facebook/bart-large-cnn')
        self.use_mask = use_mask

        self.vocab_size = 50264
        self.onehot_embeddings = nn.Linear(self.vocab_size, 1024, bias=False)
        self.onehot_embeddings.weight = nn.Parameter(self.build_lookups())

        # self.bart.encoder.embed_tokens = nn.Identity()
        # freeze bert weights
        # self.onehot_embeddings.requires_grad = False
        # self.onehot_embeddings.weight.requires_grad = False
        # for param in self.bart.parameters():
        #     param.requires_grad = False

        # Pooler
        self.l2 = torch.nn.Linear(1024, 1024)
        self.act = torch.nn.Tanh()
        # Classifier
        self.l3 = torch.nn.Dropout(0.2)
        self.l4 = torch.nn.Linear(1024, 10) # 10 categories


    def build_lookups(self):
        ids = torch.LongTensor([i for i in range(self.vocab_size)])
        return torch.transpose(self.bart.encoder.embed_tokens(ids), 0, 1).detach()
    
    def loss_fn(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def forward(self, one_hot_encodings, mask=None):
        embedded = self.onehot_embeddings(one_hot_encodings) * self.bart.encoder.embed_scale
        # embedded = self.bart.encoder.embed_positions(embedded)

        if self.use_mask:
            output_1 = self.bart.encoder(inputs_embeds=embedded, attention_mask = mask).last_hidden_state
        else:
            output_1 = self.bart.encoder(inputs_embeds=embedded).last_hidden_state

        pooled = output_1[:, 0]
        output_2 = self.act(self.l2(pooled))
        output_3 = self.l3(output_2)
        output = self.l4(output_3)
        return output
    
    def training_step(self, batch, batch_nb):
        ids = batch['ids']
        mask = batch['mask']
        y = batch['targets']
        one_hot_encodings = F.one_hot(ids, num_classes=50264).float()
        y_hat = self.forward(one_hot_encodings, mask)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        ids = batch['ids']
        mask = batch['mask']
        y = batch['targets']
        one_hot_encodings = F.one_hot(ids, num_classes=self.vocab_size).float()
        y_hat = self.forward(one_hot_encodings, mask)
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
        # ids = batch['ids']
        # mask = batch['mask']
        # y = batch['targets']
        original_ids = batch['original_ids']
        ids_with_moral_tokens = batch['ids_with_moral_tokens']

        original_mask = batch['original_mask']
        encdec_mask = batch['encdec_mask']
        target_morals = batch['target_morals']

        one_hot_encodings = F.one_hot(original_ids, num_classes=self.vocab_size).float()
        y_hat = self.forward(one_hot_encodings, original_mask)
        loss = self.loss_fn(y_hat, target_morals)
        y_preds = (y_hat >= 0).int()  
        stats =  { 'original_ids': original_ids,
                  'test_loss': loss, 
                   'progress_bar': {'test_loss': loss},
                   'y_preds': y_preds,
                   'y_hat': y_hat,
                   'y': target_morals}
        
        self.log('test_loss', loss)
        return {**stats}
    
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_preds = torch.cat([x['y_preds'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        original_ids = torch.cat([x['original_ids'] for x in outputs])

        input_text = [self.convert(tokens=t) for t in original_ids]

        y = y.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()

        for feature_idx in range(y.shape[1]):
            # print(metrics.accuracy_score(y[:, feature_idx], y_preds[:, feature_idx]), metrics.balanced_accuracy_score(y[:, feature_idx], y_preds[:, feature_idx]))
            print(metrics.f1_score(y[:, feature_idx], y_preds[:, feature_idx]))

        print("Wrong Results")
        wrong = defaultdict(list)
        for index, y_pred in enumerate(y_preds): 
            if not np.array_equal(y_pred, y[index]):
                print('Input:  {}'.format(input_text[index]))
                print('Original morals: {}'.format(', '.join(get_target_moral_names(y[index]))))
                print('Predicted morals:   {}'.format(', '.join(get_target_moral_names(y_pred))))
                print("")
                data = {"Input":input_text[index], "OG Morals": get_target_moral_names(y[index]), "Predicted Morals": get_target_moral_names(y_pred)}
                for key, value in data.items():
                    wrong[key].append(value)
        
        import pdb; pdb.set_trace()
        file_df = pd.DataFrame(wrong)
        file_df.to_csv("./classifier_wrong.csv", mode='w', header=True)

        accuracy = metrics.accuracy_score(y, y_preds)
        f1_score_micro = metrics.f1_score(y, y_preds, average='micro')
        f1_score_macro = metrics.f1_score(y, y_preds, average='macro')

        stats = {
            'acc': accuracy,
            'f1-micro': f1_score_micro,
            'f1-macro': f1_score_macro
            }

        self.log('test_loss', avg_loss)
        print(stats)
        return {**stats}

        
    def convert(self, tokens):
        sentence = tokenizer.decode(tokens)
        stop_idx = len(sentence) + 1
        if '</s>' in sentence:
            stop_idx = sentence.index('</s>')
        return sentence[3:stop_idx]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    

