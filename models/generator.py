import os
import time
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from transformers import DistilBertModel, BartModel, BartForConditionalGeneration
from transformers import BartTokenizerFast, BertTokenizerFast

# from bart_scorer import BartScorer
import logging
import transformers
from bert_score import score
import bert_score

# @inproceedings{bert-score,
#   title={BERTScore: Evaluating Text Generation with BERT},
#   author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
#   booktitle={International Conference on Learning Representations},
#   year={2020},
#   url={https://openreview.net/forum?id=SkeHuCVFDr}
# }

class BartScorer():
    def __init__(self):
        self.bart_scorer = bert_score.BERTScorer(model_type="facebook/bart-large-cnn")

    def calc_bart_score(self, candidates, references):
        # turn verbose=True on if we want status updates such as "preparing IDF dict"
        P, R, F1 = self.bart_scorer.score(candidates, references)
        return F1.mean()

from bart import MoralClassifier
from custom_transformer_classifier import OneHotMoralClassifier

# https://arxiv.org/pdf/1903.06353.pdf

START_TOK = 0
PADDING_TOK = 1
MASK_TOK = 3 # is this correct?

MASK = 0
UNMASK = 1

# Stacks moral vector with encoded representation, prior to decoder.
class MoralTransformer(pl.LightningModule):

    def __init__(self, discriminator=None, bart_decoder=True, freeze_encoder=True, freeze_decoder=True, n_contextual_linear=1, seq_len=128, moral_vec_size=10):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
        self.bart_scorer = BartScorer()

        # Load pretrained model
        # self.pretrained = BartModel.from_pretrained('facebook/bart-large-cnn')
        self.pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').cuda()

        self.encoder = self.pretrained.model.encoder
        self.embedding = self.pretrained.model.shared

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False  

        self.n_vocab = self.embedding.num_embeddings
        self.n_encoder_features = self.encoder.layernorm_embedding.normalized_shape[0]

        # Linear layer to combine encodings and moral features
        self.linears = [nn.Linear(self.n_encoder_features + moral_vec_size, self.embedding.embedding_dim)]
        for i in range(n_contextual_linear - 1):
            self.linears.append(nn.Linear(self.embedding.embedding_dim, self.embedding.embedding_dim))

        # Decoder
        if bart_decoder:
            self.decoder = self.pretrained.model.decoder
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding.embedding_dim, nhead=16, dim_feedforward=1024, dropout=0.1, activation='relu')
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.lm_head = self.pretrained.lm_head


        # self.discriminator = OneHotMoralClassifier(use_mask=False)
        # self.discriminator.load_state_dict(torch.load('../saved_models/onehot_classifier.state'))
        self.discriminator = discriminator

    def forward(self, input_seqs, input_masks, moral_targets):
        copied_morals = torch.unsqueeze(moral_targets, 1).repeat(1, self.seq_len, 1)

        encoded = self.encoder(input_seqs, input_masks).last_hidden_state
        encoded = torch.cat((encoded, copied_morals), 2)

        for linear in self.linears:
            encoded = linear(encoded)

        generated_seqs = torch.LongTensor([[PADDING_TOK] * self.seq_len] * input_seqs.shape[0]).cuda()
        generated_masks = torch.LongTensor([[MASK] * input_seqs.shape[1]] * input_seqs.shape[0]).cuda()
        decoded = self.decoder(input_ids=generated_seqs, attention_mask=generated_masks, encoder_hidden_states=encoded, encoder_attention_mask=input_masks)

        head = self.lm_head(decoded)
        outputs = F.softmax(head, dim=-1)
        # return outputs

        outputs = self.discriminator(outputs)
        return outputs

    def training_step(self, batch, batch_idx):
        input_seqs, input_masks, moral_targets, generated_seqs, generated_masks = batch
        
        generated_seqs = self.forward(input_seqs, input_masks, moral_targets, generated_seqs, generated_masks) # for discriminator and BERTSCORE
        
        # 1.  Discriminator outputs
        # TODO: how to feed this in properly
        # discriminator currently expects shape (BATCH_SIZE, seq_len) of type LongTensor (TOKENS)
        # but generator output is (BATCH_SIZE, seq_len, n_vocab) of type FloatTensor     (TOKEN PROBABILITIES)
        max_elements, max_indexes = torch.max(generated_seqs, dim=2)
        discriminator_input = max_indexes 
        predicted_morals = self.discriminator(discriminator_input) 

        # 2. BARTSCORE loss between generated_seqs and input_seqs
        content_loss = self.bart_scorer.calc_bart_score(generated_seqs, input_seqs)
        moral_loss = self.discriminator.loss_fn(predicted_morals, moral_targets)

        # 2. BERTSCORE loss between generated_seqs and input_seqs
        content_loss = 0

        # 3. Backpropagate
        loss = moral_loss + content_loss

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def decode_to_tokens(self, output):
        tokens = torch.argmax(output, 2).cpu().detach().numpy()
        results = []
        for token_set in tokens:
            converted = self.tokenizer.convert_ids_to_tokens(token_set)
            sentence = ''.join(converted).replace('Ġ', ' ')
            results.append(sentence)
        return results

if __name__ == '__main__':
    seq_len = 128
    batch_size = 16

    # transformer = MoralTransformer(seq_len=seq_len).cuda()
    transformer = MoralTransformer(seq_len=seq_len)

    input_seqs = torch.LongTensor([list(range(seq_len))] * batch_size).cuda()
    input_masks = torch.LongTensor([[UNMASK] * seq_len] * batch_size).cuda()
    moral_targets = torch.FloatTensor([list(range(10))] * batch_size).cuda()

    now = time.time()
    out_seqs = transformer(input_seqs, input_masks, moral_targets)
    elapsed = time.time() - now
    print(out_seqs.shape)

    est_train_samples = 44000
    seconds_in_hour = 60 * 60
    est_seconds_per_epoch = (est_train_samples / batch_size) * elapsed

    print('~{} secs/forward epoch'.format(round(est_seconds_per_epoch)))

    # transformer.eval()
    # outputs = transformer.forward(source, moral_target, generated)
    # for o in transformer.decode_to_tokens(outputs):
    #     print(o)

    # transformer.train()
    # loss = torch.sum(out_seqs)
    # optimizer = torch.optim.Adam(params=transformer.parameters(), lr=0.01)
    # now = time.time()
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # elapsed = time.time() - now
    # est_seconds_per_epoch = (est_train_samples / batch_size) * elapsed

    # print('~{} secs/backprop epoch'.format(round(est_seconds_per_epoch)))

    # transformer.eval()
    # outputs = transformer.forward(source, moral_target, generated)
    # for o in transformer.decode_to_tokens(outputs):
    #     print(o)

    # tokenized = transformer.tokenizer('trump biden electionk zx zc fzd vcx zx')['input_ids']
    # print(tokenized)
    # print(transformer.tokenizer.convert_ids_to_tokens(tokenized))
