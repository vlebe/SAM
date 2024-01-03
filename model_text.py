from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score

epochs = 5


data = pd.read_csv('data.csv')

data_df = pd.read_csv('data.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens = tokenizer.batch_encode_plus(data_df['text'].tolist(), max_length=256, 
                                     padding='max_length', truncation=True, 
                                     return_tensors='pt')

input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
labels = torch.tensor(data_df['turn_after'].tolist()).float()

train_features, test_features, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2)

train_dataset = TensorDataset(train_features, attention_mask[:len(train_labels)], train_labels)
test_dataset = TensorDataset(test_features, attention_mask[len(train_labels):], test_labels)




bert = BertModel.from_pretrained('bert-base-uncased')
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.bert(input_ids, attention_mask)
        output = self.classifier(output.pooler_output)
        return self.sigmoid(output)

model = BertClassifier()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs.squeeze(), labels)

        loss.backward()
        optimizer.step()

model.eval()
predictions, true_labels = [], []
for batch in test_loader:
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    predictions.append(outputs)
    true_labels.append(labels)

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")







