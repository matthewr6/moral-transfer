import torch
from bertviz import head_view
from bertviz import model_view
from transformers import BertTokenizer, BertModel


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


def sentence_saliency(sentence, model):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    embedded = torch.tensor(embed(tensor), requires_grad=True)

    # run model.dropout.eval() then model.train()
    scores = model(embedded)  # forward pass to get scores
    scores.backward()  # calculate gradient of score_max w.r.t nodes in computation graph during backward pass

    # To derive a single class saliency value for each word (i, j), take max magnitude across all embedding dimensions
    saliency, _ = torch.max(preprocess_1.grad.data.abs(), dim=2)
    return saliency
