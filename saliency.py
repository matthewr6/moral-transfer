import torch
from bertviz import head_view
from bertviz import model_view
from transformers import BertTokenizer, BertModel
from torchtext import data
import spacy
import numpy as np
import torch.nn as nn
from models import MoralTransformer
from models.custom_transformer_classifier import OneHotMoralClassifier
import matplotlib.pyplot as plt
plt.switch_backend('agg')


"""
BertViz limitations:
Works best with shorter inputs (e.g. a single sentence)
May run slowly if the input text is very long, especially for the model view.
"""


def bertviz_headview(model, tokenizer, sentence_a, sentence_b=None, layer=None, heads=None):
    """
    Call function as follows:
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentence_a, sentence_b = "the rabbit quickly hopped", "The turtle slowly crawled"
    bertviz_headview(model, tokenizer, sentence_a, sentence_b)
    """
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    if sentence_b:
        token_type_ids = inputs['token_type_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        attention = model(input_ids)[-1]
        sentence_b_start = None
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    head_view(attention, tokens, sentence_b_start, layer=layer, heads=heads)


def bertviz_modelview(model, tokenizer, sentence_a, sentence_b=None, hide_delimiter_attn=False, display_mode="dark"):
    """
    Call function as follows:
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentence_a, sentence_b = "the rabbit quickly hopped", "The turtle slowly crawled"
    bertviz_modelview(model, tokenizer, sentence_a, sentence_b, hide_delimiter_attn=False, display_mode="dark")
    """
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    if sentence_b:
        token_type_ids = inputs['token_type_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        attention = model(input_ids)[-1]
        sentence_b_start = None
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    if hide_delimiter_attn:
        for i, t in enumerate(tokens):
            if t in ("[SEP]", "[CLS]"):
                for layer_attn in attention:
                    layer_attn[0, :, i, :] = 0
                    layer_attn[0, :, :, i] = 0
    model_view(attention, tokens, sentence_b_start, display_mode=display_mode)


def sentence_saliency(sentence, model, train_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TEXT = data.Field(tokenize='spacy')
    TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)

    # Initialize hyper-parameters
    VOCAB_SIZE = len(TEXT.vocab)
    EMBEDDING_SIZE = 100
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    TEXT.vocab.vectors[0] = TEXT.vocab.vectors[1] = torch.zeros(EMBEDDING_SIZE)
    nlp = spacy.load('en')

    embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE, padding_idx=PAD_IDX).to(device)
    embed.weight.data.copy_(pretrained_embedding=TEXT.vocab.vectors)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
    embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    embedded = torch.tensor(embed(tensor), requires_grad=True)

    """run model.train() then model.dropout.eval()"""

    scores = model(embedded)  # forward pass to get scores
    scores.backward()  # calculate gradient of score_max w.r.t nodes in computation graph during backward pass

    # To derive a single class saliency value for each word (i, j), take max magnitude across all embedding dimensions
    saliency = embedded.grad.data.abs().squeeze()  # saliency, _ = torch.max(embedded.grad.data.abs(), dim=2)
    saliency_list = saliency.detach().cpu().numpy()
    return saliency_list


def plot_saliency_heatmap(sentence):
    saliency_list = sentence_saliency(sentence, model, )
    nlp = spacy.load('en')
    words = [tok for tok in nlp.tokenizer(sentence)]

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111)
    ax.set_aspect(aspect=2)
    im = plt.imshow(saliency_list, aspect='auto', interpolation='nearest', cmap=plt.cm.Blues)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Saliency", rotation=-90, va="bottom")

    ax.set_yticks(np.arange(len(words)))
    ax.set_xticks(np.arange(len(saliency_list[0]), step=20))
    ax.set_yticklabels(words)
    ax.set_title("Saliency heatmap for moral classification")

    plt.savefig('SaliencyHeatmap_' + sentence + '.pdf', format='pdf')


discriminator = OneHotMoralClassifier({}, use_mask=False)
print('Loading discriminator...')
discriminator.load_state_dict(torch.load('final_models/discriminator_titlemorals_state.pkl'))
print('Discriminator loaded')

model = MoralTransformer(discriminator=discriminator)
print('Loading generator state...')
model.load_state_dict(torch.load('final_models/special_finetuned/last.ckpt')['state_dict'])
print('Generator state loaded')
model = model.cuda()
model.dropout.eval()

plot_saliency_heatmap("Those protestors hate politicians", model, )

