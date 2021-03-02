import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from transformers import DistilBertModel, BartModel
from transformers import BartTokenizerFast, BertTokenizerFast

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
# https://github.com/jayparks/transformer/blob/master/transformer/models.py

# Stacks moral vector with encoded representation, prior to decoder.
class MoralTransformer(pl.LightningModule):

    def __init__(self, seq_len=1024, n_intermediate=512, moral_vec_size=5):
        super().__init__()
        self.seq_len = seq_len
        # self.tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')

        # Load pretrained model
        # self.pretrained = DistilBertModel.from_pretrained('distilbert-base-uncased').cuda()
        self.pretrained = BartModel.from_pretrained('facebook/bart-large-cnn')

        # self.n_vocab = self.pretrained.embeddings.word_embeddings.num_embeddings
        # self.n_encoder_features = self.pretrained.transformer.layer[-1].output_layer_norm.normalized_shape[0]
        self.n_vocab = self.pretrained.shared.num_embeddings
        self.n_encoder_features = self.pretrained.encoder.layernorm_embedding.normalized_shape[0]

        # Linear layer to combine encodings and moral features
        self.linear = nn.Linear(self.n_encoder_features + moral_vec_size, n_intermediate)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=n_intermediate, nhead=16, dim_feedforward=1024, dropout=0.1, activation='relu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
        self.decoder_head = nn.Linear(n_intermediate, self.n_vocab)

    def forward(self, source, moral_target, generated):
        copied_morals = torch.unsqueeze(moral_target, 1).repeat(1, self.seq_len, 1)

        # encoded = self.pretrained(source).last_hidden_state
        encoded = self.pretrained.encoder(source).last_hidden_state
        encoded = torch.cat((encoded, copied_morals), 2)

        encoded = self.linear(encoded)

        decoded = self.decoder(generated, encoded)
        
        head = self.decoder_head(decoded)
        return head

    def decode_to_tokens(self, output):
        tokens = torch.argmax(output, 2).cpu().detach().numpy()
        results = []
        for token_set in tokens:
            converted = self.tokenizer.convert_ids_to_tokens(token_set)
            results.append(converted)
        return results


    def training_step(self, batch, batch_idx):
        return
        # training_step defined the train loop.
        # It is independent of forward
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # # Logging to TensorBoard by default
        # self.log('train_loss', loss)
        # return loss

    def configure_optimizers(self):
        return
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer

if __name__ == '__main__':
    seq_len = 128
    batch_size = 10
    n_intermediate = 512

    transformer = MoralTransformer(seq_len=seq_len, n_intermediate=n_intermediate).cuda()

    source = torch.LongTensor([list(range(seq_len))] * batch_size).cuda()
    moral_target = torch.FloatTensor([[1,2,3,4,5]] * batch_size).cuda()
    generated = torch.FloatTensor([[list(range(n_intermediate))] * seq_len] * batch_size).cuda()
    outputs = transformer.forward(source, moral_target, generated)

    loss = torch.sum(outputs)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # tokenized = transformer.tokenizer('trump biden electionk zx zc fzd vcx zx')['input_ids']
    # print(tokenized)
    # print(transformer.tokenizer.convert_ids_to_tokens(tokenized))
