import pickle
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from operator import itemgetter
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


def left_pad(arr, maxlen=128):
    arr = np.array(arr)
    arr = arr[arr != 0]
    diff = maxlen - len(arr)
    return ([0] * diff) + arr.tolist()


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.num_unique = self.get_num_unique()
        labels = list(map(itemgetter('moral_features'), data))
        max_vals = [max(idx) for idx in zip(*labels)] 
        normalized_labels = [[val/max_vals[index] if max_vals[index] > 0 else val for index, val in enumerate(row)] for row in labels]  # moral feature wise normalization
        self.targets = [np.array([1 if i > 0 else 0 for i in row]) for row in normalized_labels]

    def __len__(self):
        return len(self.data)

    def get_num_unique(self):
        ids = np.array([a['content'][0] for a in self.data]).flatten()
        return ids.max() + 1

    def __getitem__(self, index):
        article = self.data[index]
        ids = article['content'][0]
        targets = self.targets[index]

        return {'ids': torch.tensor(left_pad(ids), dtype=torch.long), 'targets': torch.tensor(targets, dtype=torch.int)}


print("Start")
# file = open('../data/nela-covid-2020/combined/headlines_manual.pkl', 'rb')
file = open('../data/nela-covid-2020/content/headlines_contentmorals_cnn_bart_split.pkl', 'rb')
data = pickle.load(file)
data = [d for d in data if sum(d['moral_features'])]
file.close()
print("Data Loaded")

dataset = NewsDataset(data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-03  # 5
train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(train_dataset, **train_params)
testing_loader = DataLoader(test_dataset, **test_params)

print("Training Examples: " + str(train_size))
print(len(training_loader))


class MoralClassifier(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=512, embedding_dim=256):
        super(MoralClassifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, 11)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, ids):
        x = self.embeddings(ids)
        x = self.dropout(x)
        output, (h_n, c_n) = self.lstm(x)
        return self.linear(h_n[-1])


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# from torchsample.modules import ModuleTrainer
# trainer = ModuleTrainer(model)
# model = ModuleTrainer(Network())
# model.compile(loss='nll_loss', optimizer='adam')
# callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
# model.set_callbacks(callbacks)

model = MoralClassifier(dataset.num_unique)
model = model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for _, data in tqdm(enumerate(training_loader), "Training"):
        ids = data['ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % len(training_loader) == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch)

print("Training Done")


def validation():
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader), "Testing: "):
            ids = data['ids'].to(device, dtype=torch.long)
            # mask = data['mask'].to(device, dtype = torch.long)
            # token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype=torch.int)
            outputs = model(ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs)
    fin_targets = np.array(fin_targets)
    return fin_outputs, fin_targets


# validate
outputs, targets = validation()
print(outputs[:10])
print(targets[:10])
outputs[outputs >= 0.5] = 1
outputs[outputs < 0.5] = 0
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")