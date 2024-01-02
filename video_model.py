import torch
import torch.nn as nn



class VideoEmbedding(nn.Module):
    def __init__(self, pretrained_model, layers_fine_tuned = None):
        super(VideoEmbedding, self).__init__()
        self.pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.layers_fine_tuned = layers_fine_tuned
        self.set_fine_tune_settings(layers_fine_tuned)

    def set_fine_tune_settings(self, layers_fine_tuned):
        if layers_fine_tuned is None:
            for name, param in self.pretrained_model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.pretrained_model.named_parameters():
                if name in layers_fine_tuned:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def get_architecture(self):
        frozen_layers = []
        learnable_layers = []
        for name, param in self.pretrained_model.named_parameters():
            if param.requires_grad == False:
                frozen_layers.append([name, param])
            else:
                learnable_layers.append([name, param])
        with open('architecture_our_embedding_model.txt', 'w') as f :
            f.write('--------------------------FROZEN LAYERS--------------------------' + '\n')
            for layer in frozen_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')
            f.write('--------------------------LEARNABLE LAYERS--------------------------' + '\n')
            for layer in learnable_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')

    def forward(self, x):
        x = self.pretrained_model(x)
        size_dim1, size_dim2, size_dim3 = x.size(1), x.size(2), x.size(3)
        x = torch.reshape(x, (x.size(0), size_dim1 * size_dim2 * size_dim3))
        return x


class VideoModelLateFusion1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoModelLateFusion1,self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size//2)
        self.fo = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
    
    def get_architecture(self):
        learnable_layers = []
        for name, param in self.named_parameters():
            learnable_layers.append([name, param])
        with open('architecture_our_model.txt', 'w') as f :
            f.write('--------------------------LEARNABLE LAYERS--------------------------' + '\n')
            for layer in learnable_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])  # Take the last output from the GRU sequence
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fo(x)
        return x
    
class VideoModelLateFusion2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoModelLateFusion2,self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size//4)
        self.fo = nn.Linear(hidden_size//4, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)
    
    def get_architecture(self):
        learnable_layers = []
        for name, param in self.named_parameters():
            learnable_layers.append([name, param])
        with open('architecture_our_model.txt', 'w') as f :
            f.write('--------------------------LEARNABLE LAYERS--------------------------' + '\n')
            for layer in learnable_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])  # Take the last output from the GRU sequence
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fo(x)
        return x