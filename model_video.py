
import torch
import torch.nn as nn
import torch
import torch.optim as optim
from dataset import Dataset
from tqdm import tqdm


resolution = (70, 126, 3)
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = 1
output_embedding_model_shape = (32, 1280, 3, 4)
hidden_size = 512
num_layers = 1
input_size = output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]
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
        with open('architecture_our_model.txt', 'w') as f :
            f.write('--------------------------FROZEN LAYERS--------------------------' + '\n')
            for layer in frozen_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')
            f.write('--------------------------LEARNABLE LAYERS--------------------------' + '\n')
            for layer in learnable_layers:
                f.write(str(layer[0]) + '\n')
                f.write(str(layer[1]) + '\n')

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pretrained_model(x)
        size_dim1, size_dim2, size_dim3 = x.size(1), x.size(2), x.size(3)
        x = torch.reshape(x, (x.size(0), size_dim1 * size_dim2 * size_dim3))
        return x


class VideoModelLateFusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoModelLateFusion,self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
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
        x = self.fc(x[:, -1, :])  # Take the last output from the GRU sequence
        return x


embedding_model = VideoEmbedding(pretrained_model, layers_fine_tuned)    
model = VideoModelLateFusion(input_size, hidden_size, num_layers, num_classes)
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
indice1 = torch.randperm(1116)
indice2 = torch.randperm(len(dataset)- 1984) + 1984
indice = torch.cat((indice1,indice2))

train_index = indice[:int(len(dataset)*0.8)//batch_size*batch_size]
test_index = indice[int(len(dataset)*0.8)//batch_size*batch_size:]
train_dataset = torch.utils.data.Subset(dataset, train_index)
test_dataset = torch.utils.data.Subset(dataset, test_index)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

def train(embedding_model, model, train_loader, optimizer, criterion):
    print("Training...")
    model.train()
    running_loss = 0.0
    for i, (labels, _, frames) in tqdm(enumerate(train_loader)):
        labels = 1.0 - 2*labels.view(-1, 1).float()
        optimizer.zero_grad()
        sequence = torch.zeros((output_embedding_model_shape[0],4, output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]))
        for j in range(frames.shape[1]):
            sequence[:,j, :] = embedding_model(frames[:, j, :, :, :])
        outputs = model(sequence)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(embedding_model, model, test_loader, criterion):
    print("Testing...")
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (labels, _, frames) in tqdm(enumerate(test_loader)):
            labels = 1.0 - 2*labels.view(-1, 1).float()
            sequence = torch.zeros((1,4, output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]))
            for j in range(frames.shape[1]):
                sequence[:,j, :] = embedding_model(frames[:,j, :, :, :])
            outputs = model(sequence)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(test_loader)

for epoch in range(num_epochs):
    train_loss = train(embedding_model, model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1} : train loss {train_loss}")


torch.save(embedding_model.state_dict(), 'embedding_model.pt')
torch.save(model.state_dict(), 'model.pt')

# model = VideoModelLateFusion(input_size, hidden_size, num_layers, num_classes)
# model.load_state_dict(torch.load('model.pt'))
# embedding_model = VideoEmbedding(pretrained_model, layers_fine_tuned)
# embedding_model.load_state_dict(torch.load('embedding_model.pt'))
test_loss = test(embedding_model, model, test_loader, criterion)
print(f"Test loss {test_loss}")





