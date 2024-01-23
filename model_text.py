
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm




class DistilCamembertEmbedding(nn.Module):
    def __init__(self):
        super(DistilCamembertEmbedding, self).__init__()
        self.embedding_model = AutoModel.from_pretrained("cmarkea/distilcamembert-base")

    def forward(self, input_ids, attention_mask):
        self.embedding_model.eval()
        output = self.embedding_model(input_ids, attention_mask)
        return output.last_hidden_state[:, 0, :]
    
    def get_architecture(self):
        frozen_layers = []
        learnable_layers = []
        with open('embedding_text_model.txt', 'w') as f :
            for name, param in self.embedding_model.named_parameters():
                f.write('--------------------------LAYERS--------------------------' + '\n')
                f.write(str(name) + '\n')

class FlaubertEmbedding(nn.Module):
    def __init__(self):
        super(FlaubertEmbedding, self).__init__()
        self.embedding_model = AutoModel.from_pretrained("flaubert/flaubert_base_cased")

    def forward(self, input_ids, attention_mask):
        self.embedding_model.eval()
        output = self.embedding_model(input_ids, attention_mask)
        return output.last_hidden_state[:, 0, :]
    
class TextModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextModel, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size//8)
        self.batchnorm1 = nn.BatchNorm1d(input_size//8)
        self.fo = nn.Linear(input_size//8, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fo(x)
        return x
    
    
class TextModel2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextModel2, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size*2)
        self.batchnorm1 = nn.BatchNorm1d(input_size*2)
        self.fc2 = nn.Linear(input_size*2, input_size//16)
        self.batchnorm2 = nn.BatchNorm1d(input_size//16)
        self.fo = nn.Linear(input_size//16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fo(x)
        return x

class TextModel3(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextModel3, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size*2)
        self.batchnorm1 = nn.BatchNorm1d(input_size*2)
        self.fc2 = nn.Linear(input_size*2, input_size*2)
        self.batchnorm2 = nn.BatchNorm1d(input_size*2)
        self.fc3 = nn.Linear(input_size*2, input_size//8)
        self.batchnorm3 = nn.BatchNorm1d(input_size//8)
        self.fc4 = nn.Linear(input_size//8, input_size//16)
        self.batchnorm4 = nn.BatchNorm1d(input_size//16)
        self.fo = nn.Linear(input_size//16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fo(x)
        return x
    

if __name__ == "__main__":
    print("Bonjour je suis un test")
    txt = "Bonjour je suis un test"
    model = FlaubertEmbedding()
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")
    model_2 = TextModel(768, 2)
    for i in tqdm(range(100)):
        model.eval()
        model_2.eval()
        tokenizer_output = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = tokenizer_output["input_ids"], tokenizer_output["attention_mask"]
        embedding = model(input_ids, attention_mask)
        output = model_2(embedding)
    pass








