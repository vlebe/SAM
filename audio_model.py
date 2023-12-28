import torch.nn as nn
import torch
from vggish_model import VGGish

class AudioLSTMModel(nn.Module):
    def __init__(self):
        super(AudioLSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=20, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, x = self.lstm(x)
        x = x[0].permute(1, 0, 2)
        x = self.flatten(x)
        x = self.sigmoid(self.linear(x)).view(-1)
        return x
    
class AudioMLPModel(nn.Module):
    def __init__(self):
        super(AudioMLPModel, self).__init__()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(3960, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        x = self.sigmoid(x).view(-1)
        return x
    
class AudioEndToEndModel(nn.Module) :
    def __init__(self) :
        super().__init__()

        self.embedding = VGGish()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, x) :
        x = self.relu(self.embedding(x))
        x = self.relu(self.l1(x))
        x = self.sigmoid(self.l2(x))

        return x.view(-1)

if __name__ == '__main__':
    model = AudioEndToEndModel()
    x = torch.randn(2, 1, 96, 64)
    output = model(x)
    print(output.shape)
    print(output)