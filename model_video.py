
import torch
import torch.nn as nn
import torch
import torch.optim as optim
from dataset import Dataset


resolution = (70, 126, 3)
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = 2
hidden_size = 512
num_layers = 2
input_size = 512
learning_rate_fine_tuned_layers = 0.0001  
learning_rate_default = 0.001  



pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
total_nb_pt_layers = len(pretrained_model.features)
nb_pt_layers_to_fine_tune = total_nb_pt_layers - 0

if nb_pt_layers_to_fine_tune == total_nb_pt_layers:
    layers_fine_tuned = None
    layers_fine_tuned_name = None
else:
    layers_fine_tuned = pretrained_model.features[nb_pt_layers_to_fine_tune:]
    layers_fine_tuned_name = [name for name, param in layers_fine_tuned.named_parameters()]



class VideoModelLateFusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, pretrained_model, layers_fine_tuned = None):
        super(VideoModelLateFusion,self).__init__()
            # MobileNetV2 features (excluding classifier)
        self.mobilenetv2_features = nn.Sequential(*list(pretrained_model.children())[:-1])
        
        # GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.layers_fine_tuned = layers_fine_tuned
        self.set_fine_tune_settings(layers_fine_tuned)
    
    def set_fine_tune_settings(self, layers_fine_tuned):
        if layers_fine_tuned is None:
            for name, param in self.mobilenetv2_features.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.mobilenetv2_features.named_parameters():
                if name in layers_fine_tuned:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
    def forward(self, x):
        # MobileNetV2 features
        x = self.mobilenetv2_features(x)
        
        # Reshape for GRU input
        x = x.view(x.size(0), -1, x.size(1))
        
        # GRU layer
        x, _ = self.gru(x)
        
        # Fully connected layer
        x = self.fc(x[:, -1, :])  # Take the last output from the GRU sequence
        
        return x
    
    def get_architecture(self):
        frozen_layers = []
        learnable_layers = []
        for name, param in self.mobilenetv2_features.named_parameters():
            if param.requires_grad == False:
                frozen_layers.append([name, param])
            else:
                learnable_layers.append([name, param])
        with open('architecture_our_model.txt', 'w') as f :
            f.write('--------------------------FROZEN LAYERS--------------------------' + '\n')
            for layer in frozen_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')
            f.write('--------------------------LEARNABLE LAYERS--------------------------' + '\n')
            for layer in learnable_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')


    
model = VideoModelLateFusion(input_size, hidden_size, num_layers, num_classes, pretrained_model, layers_fine_tuned)
model.get_architecture()

# Specify the parameter groups and learning rates
params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if layers_fine_tuned is None:
            params.append({'params': param, 'lr': learning_rate_default})
        else:
            if name is not None and name in layers_fine_tuned_name:
                params.append({'params': param, 'lr': learning_rate_fine_tuned_layers})
            else:
                params.append({'params': param, 'lr': learning_rate_default})

optimizer = optim.Adam(params=params)
criterion = nn.BCEWithLogitsLoss()

dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', resolution)
indice = torch.randperm(len(dataset))
train_index = indice[:int(len(dataset)*0.8)]
test_index = indice[int(len(dataset)*0.8):]
train_dataset = torch.utils.data.Subset(dataset, train_index)
test_dataset = torch.utils.data.Subset(dataset, test_index)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, (labels, txt, audio, frames) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        input("Press Enter to continue...")
    return running_loss / len(train_loader)

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (labels, txt, audio, frames) in enumerate(test_loader):
            outputs = model(frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(test_loader)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = test(model, test_loader, criterion)
    print(f"Epoch {epoch+1} : train loss {train_loss}, test loss {test_loss}")

torch.save(model.state_dict(), 'model.pt')



