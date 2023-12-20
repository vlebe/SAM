import torch.nn as nn
import torch

class AudioLSTMModel(nn.Module):
    def __init__(self):
        super(AudioLSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=20, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 1)
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

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1300, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(self.linear3(x)).view(-1)
        return x
    
if __name__ == '__main__':
    model = AudioLSTMModel()
    x = torch.randn(2, 100, 13)
    h_out = model(x)
    print(h_out.shape)