import os
import time
import sys
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from transformers import DistilBertModel, BartModel, BartForConditionalGeneration
from transformers import BartTokenizerFast, BertTokenizerFast
# from custom_transformer_classifier import OneHotMoralClassifier
# from bart_scorer import BartScorer
import logging
import transformers
from bert_score import score
import bert_score

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

class BartScorer():
    def __init__(self):
        self.bart_scorer = bert_score.BERTScorer(model_type="facebook/bart-large-cnn")

    def calc_bart_score(self, candidates, references):
        # turn verbose=True on if we want status updates such as "preparing IDF dict"
        P, R, F1 = self.bart_scorer.score(candidates, references)
        return F1.mean()

# https://arxiv.org/pdf/1903.06353.pdf

START_TOK = 0
PADDING_TOK = 1
MASK_TOK = 3 # is this correct?

MASK = 0
UNMASK = 1

# Stacks moral vector with encoded representation, prior to decoder.
class MoralTransformerSpecial(pl.LightningModule):

    def __init__(self, 
                lr=0.001, discriminator=None, bart_decoder=True, 
                freeze_encoder=True, freeze_decoder=True, contextual_injection=True, 
                n_contextual_linear=2, moral_vec_size=10, 
                use_content_loss=False, content_loss_type='cosine',
                feed_moral_tokens_to='encoder', use_moral_loss=False):
        super().__init__()
        assert n_contextual_linear >= 1
        self.lr = lr
        self.contextual_injection = contextual_injection
        self.feed_moral_tokens_to = feed_moral_tokens_to
        self.use_moral_loss = use_moral_loss
        self.use_content_loss = use_content_loss
        self.content_loss_type = content_loss_type

        self.loss_history = []
        self.training_epoch_count = 0
        self.use_original_morals = True 
        
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
        self.bart_scorer = BartScorer()

        # Load pretrained model
        # self.pretrained = BartModel.from_pretrained('facebook/bart-large-cnn')
        print('Loading pretrained bart-large-cnn...')
        self.pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
        print('Pretrained bart-large-cnn loaded')
        # print(self.pretrained)
        # sys.exit()

        self.encoder = self.pretrained.model.encoder
        self.embedding = self.pretrained.model.shared

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False  

        self.n_vocab = self.embedding.num_embeddings
        self.n_encoder_features = self.encoder.layernorm_embedding.normalized_shape[0]

        # Linear layers to combine encodings and moral features
        self.linears = nn.ModuleList([nn.Linear(self.n_encoder_features + moral_vec_size, self.embedding.embedding_dim)])
        for i in range(n_contextual_linear - 1):
            self.linears.append(nn.Linear(self.embedding.embedding_dim, self.embedding.embedding_dim))

        # Decoder
        self.decoder = self.pretrained.model.decoder
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.lm_head = self.pretrained.lm_head

        self.discriminator = discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False  

        self.vocab_size = 50264
        self.onehot_embeddings = nn.Linear(self.vocab_size, 1024, bias=False)
        self.onehot_embeddings.weight = nn.Parameter(self.discriminator.build_lookups())
        self.onehot_embeddings.requires_grad = False
        self.onehot_embeddings.weight.requires_grad = False

    def forward(self, encoder_seqs, decoder_seqs, encoder_mask, decoder_mask, moral_targets):
        batch_size = encoder_seqs.shape[0]
        seq_len = encoder_seqs.shape[1]
        copied_morals = torch.unsqueeze(moral_targets, 1).repeat(1, seq_len, 1)

        encoded = self.encoder(encoder_seqs, encoder_mask).last_hidden_state

        if self.contextual_injection:
            encoded = torch.cat((encoded, copied_morals), 2)
            for linear in self.linears:
                encoded = linear(encoded)

        decoded = self.decoder(input_ids=decoder_seqs, attention_mask=decoder_mask, encoder_hidden_states=encoded, encoder_attention_mask=encoder_mask).last_hidden_state

        outputs = self.lm_head(decoded)
        if self.training_epoch_count >= 10: 
            outputs = F.softmax(outputs, dim=-1)
        return outputs

    def loss_fn(self, input_seqs, generated_seqs, moral_targets, predicted_morals): 
        # 1. Moral loss
        if self.use_moral_loss:
            moral_loss = self.discriminator.loss_fn(predicted_morals, moral_targets)
        else:
            moral_loss = 0

        # 2. Content loss between generated_seqs and input_seqs
        if self.use_content_loss:
            # different loss for first 10 epochs
            if self.training_epoch_count < 10: 
                input_seqs = F.one_hot(input_seqs, num_classes=50264).float()
                content_loss = nn.CrossEntropyLoss()(generated_seqs, input_seqs)
           else:
                input_embeddings = self.encoder(input_seqs).last_hidden_state
                input_embeddings = torch.mean(input_embeddings, 1)

                output_embeddings = self.onehot_embeddings(generated_seqs)
                output_embeddings = self.encoder(inputs_embeds=output_embeddings).last_hidden_state
                output_embeddings = torch.mean(output_embeddings, 1)
                
                if self.content_loss_type == 'cosine':
                    content_loss = F.cosine_similarity(input_embeddings, output_embeddings)
                elif self.content_loss_type == 'pairwise': 
                    content_loss = F.pairwise_distance(input_embeddings, output_embeddings)
                elif self.content_loss_type == "normalized_pairwise": # normalized Euclidean Distance
                    unit_input = F.normalize(input_embeddings)
                    unit_output = F.normalize(output_embeddings)
                    content_loss = F.pairwise_distance(unit_input, unit_output) / 2
                content_loss = torch.mean(content_loss)

                
        else:
            content_loss = 0

        # 3. Final loss
        loss = moral_loss + content_loss
        return loss

    def training_step(self, batch, batch_idx):
        original_ids = batch['original_ids']
        ids_with_moral_tokens = batch['ids_with_moral_tokens']

        original_mask = batch['original_mask']
        encdec_mask = batch['encdec_mask']
        target_morals = batch['target_morals']

        import pdb; pdb.set_trace()

        # encoder_seqs, decoder_seqs, encoder_mask, decoder_mask, moral_targets
        if self.feed_moral_tokens_to == 'encoder':
            generated_seqs = self.forward(ids_with_moral_tokens, original_ids, encdec_mask, original_mask, target_morals)
        else:
            generated_seqs = self.forward(original_ids, ids_with_moral_tokens, original_mask, encdec_mask, target_morals)

        predicted_morals = self.discriminator(generated_seqs) 
        

        if self.use_original_morals:
            original_morals = batch['original_morals']
            loss = self.loss_fn(original_ids, generated_seqs, original_morals, predicted_morals)
        else: 
            loss = self.loss_fn(original_ids, generated_seqs, target_morals, predicted_morals)


        self.log('train_loss', loss)
        self.loss_history.append(loss.item())
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        import pdb; pdb.set_trace()
        self.training_epoch_count += 1
        if self.training_epoch_count >= 10: 
            self.use_moral_loss = True
            self.use_original_morals = False




    def validation_step(self, batch, batch_nb):
        original_ids = batch['original_ids']
        ids_with_moral_tokens = batch['ids_with_moral_tokens']

        original_mask = batch['original_mask']
        encdec_mask = batch['encdec_mask']
        target_morals = batch['target_morals']

        # encoder_seqs, decoder_seqs, encoder_mask, decoder_mask, moral_targets
        if self.feed_moral_tokens_to == 'encoder':
            generated_seqs = self.forward(ids_with_moral_tokens, original_ids, encdec_mask, original_mask, target_morals)
        else:
            generated_seqs = self.forward(original_ids, ids_with_moral_tokens, original_mask, encdec_mask, target_morals)

        predicted_morals = self.discriminator(generated_seqs)

        loss = self.loss_fn(original_ids, generated_seqs, target_morals, predicted_morals)
        predicted_morals_preds = (predicted_morals >= 0).int()  
        stats =  {'val_loss': loss, 
                   'progress_bar': {'val_loss': loss},
                   'predicted_morals_preds': predicted_morals_preds,
                   'predicted_morals': predicted_morals,
                   'target_morals': target_morals}
        
        self.log('val_loss', loss)
        return {**stats}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        target_morals = torch.cat([x['target_morals'] for x in outputs])
        predicted_morals_preds = torch.cat([x['predicted_morals_preds'] for x in outputs])
        predicted_morals = torch.cat([x['predicted_morals'] for x in outputs])

        accuracy = metrics.accuracy_score(target_morals.cpu(), predicted_morals_preds.cpu())
        f1_score_micro = metrics.f1_score(target_morals.cpu(), predicted_morals_preds.cpu(), average='micro')
        f1_score_macro = metrics.f1_score(target_morals.cpu(), predicted_morals_preds.cpu(), average='macro')

        stats = {
            'acc': accuracy,
            'f1-micro': f1_score_micro,
            'f1-macro': f1_score_macro
            }
        
        self.log('val_loss', avg_loss)
        print(stats)
        return {**stats}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


    discriminator = OneHotMoralClassifier({}, use_mask=False)
    discriminator.load_state_dict(torch.load('discriminator_state.pkl', map_location=device))
    transformer = MoralTransformer(discriminator=discriminator, content_loss_type="normalized_pairwise").to(device)

    input_seqs = torch.LongTensor([list(range(seq_len))] * batch_size).to(device)
    input_masks = torch.LongTensor([[UNMASK] * seq_len] * batch_size).to(device)
    moral_targets = torch.FloatTensor([list(range(10))] * batch_size).to(device)

    transformer.training_step(input_seqs, input_masks, moral_targets)

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

    transformer.train()
    loss = torch.sum(out_seqs)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=0.001)
    now = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - now
    est_seconds_per_epoch = (est_train_samples / batch_size) * elapsed

    # print('~{} secs/backprop epoch'.format(round(est_seconds_per_epoch)))

    # transformer.eval()
    # outputs = transformer.forward(source, moral_target, generated)
    # for o in transformer.decode_to_tokens(outputs):
    #     print(o)

    # tokenized = transformer.tokenizer('trump biden electionk zx zc fzd vcx zx')['input_ids']
    # print(tokenized)
    # print(transformer.tokenizer.convert_ids_to_tokens(tokenized))
