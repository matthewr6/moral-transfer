import torch


# pre-process the sentence into vec
def pre_process(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    embedded = torch.tensor(embed(tensor), requires_grad=True)
    return embedded


# deal with input sentence
input_1 = u"This film is horrible!"
input_1 = u"This movie was sadly under-promoted but proved to be truly exceptional."
preprocess_1 = pre_process(input_1)  # requires_grad = True

# model.dropout.eval()
# model.train()

# forward pass to get scores
scores = model(preprocess_1)

# calculate gradient of score_max w.r.t the nodes in the computation graph during backward pass
scores.backward()

# To derive a single class saliency value for each word (i, j),  we take the maximum magnitude across all embedding dimensions.
saliency, _ = torch.max(preprocess_1.grad.data.abs(), dim=2)
