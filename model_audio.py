import torch.nn as nn
import torch

class AudioRNNModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=512, num_layers=1, num_classes=2):
        super(AudioRNNModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(hidden_size, hidden_size*2)
        self.linear3 = nn.Linear(hidden_size*2, hidden_size//2)
        self.linear4 = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size*2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size*2)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size//2)

        
    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.dropout2(self.relu(self.batchnorm1(self.linear1(x))))
        x = self.dropout1(self.relu(self.batchnorm3(self.linear3(x))))
        x = self.linear4(x)
        return x
    
class AudioMLPModel1(nn.Module):
    def __init__(self, input_size=3960,num_classes=2):
        super(AudioMLPModel1, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size//16)
        self.linear2 = nn.Linear(input_size//16, input_size//64)
        self.fco = nn.Linear(input_size//64, num_classes)
        self.batchnorm1 = nn.BatchNorm1d(input_size//16)
        self.batchnorm2 = nn.BatchNorm1d(input_size//64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.batchnorm1(self.linear1(x))))
        x = self.dropout(self.relu(self.batchnorm2(self.linear2(x))))
        x = self.fco(x)
        return x

class AudioMLPModel2(nn.Module):
    def __init__(self):
        super(AudioMLPModel2, self).__init__()
        self.linear1 = nn.Linear(3960, 3960*2)
        self.linear3 = nn.Linear(3960*2, 3960*2)
        self.linear4 = nn.Linear(3960*2, 3960//16)
        self.linear5 = nn.Linear(3960//16, 3960//64)
        self.fco = nn.Linear(3960//64, 2)
        self.batchnorm1 = nn.BatchNorm1d(3960*2)
        self.batchnorm3 = nn.BatchNorm1d(3960*2)
        self.batchnorm4 = nn.BatchNorm1d(3960//16)
        self.batchnorm5 = nn.BatchNorm1d(3960//64)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.9)
        self.dropout2 = nn.Dropout(0.7)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout4(self.relu(self.batchnorm1(self.linear1(x))))
        x = self.dropout3(self.relu(self.batchnorm3(self.linear3(x))))
        x = self.dropout2(self.relu(self.batchnorm4(self.linear4(x))))
        x = self.dropout1(self.relu(self.batchnorm5(self.linear5(x))))
        x = self.fco(x)
        return x
    
class AudioMLPModel3(nn.Module):
    def __init__(self):
        super(AudioMLPModel3, self).__init__()
        self.linear1 = nn.Linear(3960, 3960*2)
        self.linear2 = nn.Linear(3960*2, 3960*2)
        self.linear3 = nn.Linear(3960*2, 3960*2)
        self.linear4 = nn.Linear(3960*2, 3960)
        self.linear5 = nn.Linear(3960, 3960//16)
        self.linear6 = nn.Linear(3960//16, 3960//64)
        self.fco = nn.Linear(3960//64, 2)
        self.batchnorm1 = nn.BatchNorm1d(3960*2)
        self.batchnorm2 = nn.BatchNorm1d(3960*2)
        self.batchnorm3 = nn.BatchNorm1d(3960*2)
        self.batchnorm4 = nn.BatchNorm1d(3960)
        self.batchnorm5 = nn.BatchNorm1d(3960//16)
        self.batchnorm6 = nn.BatchNorm1d(3960//64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        input = x
        x = self.dropout(self.relu(self.batchnorm1(self.linear1(input))))
        x = self.dropout(self.relu(self.batchnorm2(self.linear2(x))))
        x = self.dropout(self.relu(self.batchnorm3(self.linear3(x))))
        x = self.dropout(self.relu(self.batchnorm4(self.linear4(x))))
        x = self.dropout(self.relu(self.batchnorm5(self.linear5((x + input)/2))))
        x = self.dropout(self.relu(self.batchnorm6(self.linear6(x))))
        x = self.fco(x)
        return x

if __name__ == '__main__':
    model = AudioMLPModel3()
    print(model)
    x = torch.randn(1, 3960)
    print(model(x))