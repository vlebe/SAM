
from transformers import AutoModel
import torch
import torch.nn as nn
from transformers import pipeline




class DistilCamembertEmbedding(nn.Module):
    def __init__(self):
        super(DistilCamembertEmbedding, self).__init__()
        self.embedding_model = AutoModel.from_pretrained("cmarkea/distilcamembert-base")

    def forward(self, input_ids, attention_mask):
        self.embedding_model.eval()
        output = self.embedding_model(input_ids, attention_mask)
        return output.last_hidden_state[:, 0, :]
        # return output.pooler_output
    
class TextModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(TextModel, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size//8)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size//8)
        self.fo = nn.Linear(hidden_size//8, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fo(x)
        return x
    

if __name__ == "__main__":
    embedding_model = DistilCamembertEmbedding()
    input = "Bonjour, je m'appelle Camille."
    print(embedding_model(input).size())
    hidden_size = 768
    model = TextModel(hidden_size, 2)
    model.eval()
    print(torch.nn.functional.softmax(model(embedding_model(input)), dim=1))








