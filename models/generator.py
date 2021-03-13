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

from transformers import DistilBertModel, BartModel
from transformers import BartTokenizerFast, BertTokenizerFast

from bart import MoralClassifier

# https://arxiv.org/pdf/1903.06353.pdf

START_TOK = 0
PADDING_TOK = 1
MASK_TOK = 3 # is this correct?

# Stacks moral vector with encoded representation, prior to decoder.
class MoralTransformer(pl.LightningModule):

    def __init__(self, seq_len=128, moral_vec_size=5, discriminator=None):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')

        # Load pretrained model
        self.pretrained = BartModel.from_pretrained('facebook/bart-large-cnn')

        self.encoder = self.pretrained.encoder
        self.encoder.requires_grad = False
        self.embedding = self.pretrained.shared

        self.n_vocab = self.embedding.num_embeddings
        self.n_encoder_features = self.encoder.layernorm_embedding.normalized_shape[0]

        # Linear layer to combine encodings and moral features
        self.linear = nn.Linear(self.n_encoder_features + moral_vec_size, self.embedding.embedding_dim)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding.embedding_dim, nhead=16, dim_feedforward=1024, dropout=0.1, activation='relu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
        self.decoder_head = nn.Linear(self.embedding.embedding_dim, self.n_vocab)

        # self.discriminator = discriminator

        # checkpoint = torch.load('../saved_models/classifier_frozen_encoder_lr_1e-4.ckpt')
        self.discriminator = MoralClassifier.load_from_checkpoint('../saved_models/classifier_frozen_encoder_lr_1e-4.ckpt')
        # self.discriminator.load_state_dict(checkpoint['state_dict'])

        self.to_discrim_input = nn.Linear(self.n_vocab, 1) # temporary argmax hack

    def build_onehot_embeddings(self):
        ids = torch.LongTensor([i for i in range(self.vocab_size)])
        return torch.transpose(self.embedding(ids), 0, 1).detach()

    def forward(self, input_seqs, input_masks, moral_targets, generated_seqs, generated_masks): # create genrated seqs and mask instead to just mask everything??
        
        copied_morals = torch.unsqueeze(moral_targets, 1).repeat(1, self.seq_len, 1)

        encoded = self.encoder(input_seqs, input_masks).last_hidden_state
        encoded = torch.cat((encoded, copied_morals), 2)

        encoded = self.linear(encoded)

        generated_embeddings = self.embedding(generated_seqs)
        decoded = self.decoder(generated_embeddings, encoded, input_masks, generated_masks)

        head = self.decoder_head(decoded)
        outputs = F.softmax(head, dim=-1)
        return outputs

        # outputs = self.to_discrim_input(outputs)
        # outputs = torch.squeeze(outputs, -1)

        # outputs = self.discriminator(outputs.long())

        # return outputs

    def training_step(self, batch, batch_idx):
        input_seqs, input_masks, moral_targets, generated_seqs, generated_masks = batch
        
        generated_seqs = self(input_seqs, input_masks, moral_targets, generated_seqs, generated_masks) # for discriminator and BERTSCORE
        
        # 1.  Discriminator outputs
        # TODO: how to feed this in properly
        # discriminator currently expects shape (BATCH_SIZE, seq_len) of type LongTensor (TOKENS)
        # but generator output is (BATCH_SIZE, seq_len, n_vocab) of type FloatTensor     (TOKEN PROBABILITIES)
        predicted_morals = self.discriminator(generated_seqs)

        # 2. BERTSCORE loss between generated_seqs and input_seqs

        # 3. Backpropagate through discriminator: BCEWithLogitsLoss(predicted_morals, moral_targets)

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

    transformer = MoralTransformer(seq_len=seq_len).cuda()

    input_seqs = torch.LongTensor([list(range(seq_len))] * batch_size).cuda()
    moral_targets = torch.FloatTensor([[1,2,3,4,5]] * batch_size).cuda()
    generated_seqs = torch.LongTensor([[1] * seq_len] * batch_size).cuda()

    now = time.time()
    out_seqs = transformer(input_seqs, moral_targets, generated_seqs)
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

    transformer.train()
    loss = torch.sum(out_seqs)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=0.01)
    now = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - now
    est_seconds_per_epoch = (est_train_samples / batch_size) * elapsed

    print('~{} secs/backprop epoch'.format(round(est_seconds_per_epoch)))

    # transformer.eval()
    # outputs = transformer.forward(source, moral_target, generated)
    # for o in transformer.decode_to_tokens(outputs):
    #     print(o)

    # tokenized = transformer.tokenizer('trump biden electionk zx zc fzd vcx zx')['input_ids']
    # print(tokenized)
    # print(transformer.tokenizer.convert_ids_to_tokens(tokenized))
