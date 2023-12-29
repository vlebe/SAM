
import torch
import torch.nn as nn
import torch
import torch.optim as optim
from dataset import Dataset
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
import torchvision 

    

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


class VideoModelLateFusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoModelLateFusion,self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size//2)
        # self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        # self.batchnorm2 = nn.BatchNorm1d(hidden_size//4)
        self.fo = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        # self.dropout2 = nn.Dropout(p=0.5)
    
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
        # x = self.fc2(x)
        # x = self.batchnorm2(x)
        # x = self.relu(x)
        # x = self.dropout2(x)
        x = self.fo(x)
        return x

def train(embedding_model, model, train_loader, optimizer, criterion, output_embedding_model_shape):

    print("Training...")
    embedding_model.train()
    model.train()
    running_loss = 0.0
    for i, (labels, _, frames) in tqdm(enumerate(train_loader)):
        labels = 1.0 - 2*labels.view(-1, 1).float()
        sequence = torch.zeros((output_embedding_model_shape[0],4, output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]))
        if torch.backends.mps.is_available():
            labels = labels.to('mps')
            frames = frames.to('mps')
            sequence = sequence.to('mps')
        optimizer.zero_grad()
        for j in range(frames.shape[1]):
            sequence[:,j, :] = embedding_model(frames[:, j, :, :, :])
        outputs = model(sequence)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(embedding_model, model, test_loader, criterion, output_embedding_model_shape, threshold=0.7):
    global device
    print("Testing...")
    embedding_model.eval()
    model.eval()
    running_loss = 0.0
    total_predictions = torch.tensor([], dtype=torch.float32, device = device)
    total_labels = torch.tensor([], dtype=torch.float32, device = device)

    with torch.no_grad():
        for i, (labels, _, frames) in tqdm(enumerate(test_loader)):
            sequence = torch.zeros((1, 4, output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]), dtype=torch.float32)
            
            if torch.backends.mps.is_available():
                labels = labels.to('mps')
                frames = frames.to('mps')
                sequence = sequence.to('mps')
            
            labels = 1.0 - 2 * labels.view(-1, 1).float()

            for j in range(frames.shape[1]):
                sequence[:, j, :] = embedding_model(frames[:, j, :, :, :])

            outputs = model(sequence)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Concatenate predictions and labels for later metric calculation
            total_predictions = torch.cat((total_predictions, outputs.view(-1)))
            total_labels = torch.cat((total_labels, labels.view(-1)))

            # Calculate precision, recall, and accuracy for the current batch
            binary_predictions = torch.argmax(torch.sigmoid(outputs), dim=1)
            tp = torch.sum((binary_predictions == 1) & (labels == 1)).item()
            fp = torch.sum((binary_predictions == 1) & (labels == 0)).item()
            tn = torch.sum((binary_predictions == 0) & (labels == 0)).item()
            fn = torch.sum((binary_predictions == 0) & (labels == 1)).item()


    average_loss = running_loss / len(test_loader)
    thresholded_predictions = torch.where(torch.sigmoid(total_predictions) > threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    # Calculate overall precision, recall, and accuracy
    overall_tp = torch.sum((thresholded_predictions == 1) & (total_labels == 1)).item()
    overall_fp = torch.sum((thresholded_predictions == 1) & (total_labels == 0)).item()
    overall_tn = torch.sum((thresholded_predictions == 0) & (total_labels == 0)).item()
    overall_fn = torch.sum((thresholded_predictions == 0) & (total_labels == 1)).item()

    overall_precision = overall_tp / max(overall_tp + overall_fp, 1e-10)
    overall_recall = overall_tp / max(overall_tp + overall_fn, 1e-10)
    overall_accuracy = (overall_tp + overall_tn) / max(overall_tp + overall_tn + overall_fp + overall_fn, 1e-10)
    overall_f1_score = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-10)

    print(f'Overall Metrics - Loss: {average_loss:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, Accuracy: {overall_accuracy:.4f} F1 Score: {overall_f1_score:.4f}')
    print(f"Test loss {average_loss}")







if __name__ == '__main__':

    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--learning_rate_default', type=float, default=0.0001, help='learning rate for other layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--max_iter', type=int, default=10, help='maximum number of iteration, witout any improvement on the train loss with tol equal to 1e-3')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(32, 2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing. 6 is recommended if available.')
    args = parser.parse_args()
    batch_size = args.batch_size
    input_size = args.output_embedding_model_shape[1] * args.output_embedding_model_shape[2] * args.output_embedding_model_shape[3]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device {device}.")

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    pretrained_model = torchvision.models.resnet50(weights=weights)
    pretrained_model = pretrained_model.to(device) 
    preprocess = weights.transforms().to(device)

    embedding_model = VideoEmbedding(pretrained_model)
    model = VideoModelLateFusion(input_size, args.hidden_size, args.num_layers, args.num_classes)
    embedding_model.to(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)

    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', args.resolution, preprocess)
    indice1 = torch.randperm(1116)
    indice2 = torch.randperm(len(dataset)- 1984) + 1984
    indice = torch.cat((indice1,indice2))
    train_index = indice[:int(len(dataset)*0.8)//batch_size*batch_size]
    test_index = indice[int(len(dataset)*0.8)//batch_size*batch_size:]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    if device == torch.device('cuda') or device == torch.device('mps'):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    # for batch in train_loader:
    #     labels, _, frames = batch
    #     plt.imshow(frames[0, 0, :, :, :].permute(1, 2, 0))
    #     plt.show()
    #     break

    print("Start training...")
    old_train_loss = 0
    iteration = 0
    for epoch in range(args.num_epochs):
        train_loss = train(embedding_model, model, train_loader, optimizer, criterion, args.output_embedding_model_shape)
        if old_train_loss - train_loss < 1e-3:
            iteration += 1
        else:
            iteration = 0
        if iteration == args.max_iter or old_train_loss - train_loss < 0.0 :
            print(f"Early stopping at epoch {epoch+1} : train loss {train_loss}")
            break
        print(f"Epoch {epoch+1} : train loss {train_loss}")


    torch.save(embedding_model.state_dict(), 'data/parameter_models/embedding_modelV3.pt')
    torch.save(model.state_dict(), 'data/parameter_models/modelV3.pt')

    # model = VideoModelLateFusion(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
    # model.load_state_dict(torch.load('data/parameter_models/modelV3.pt'))
    # embedding_model = VideoEmbedding(pretrained_model, layers_fine_tuned).to(device)
    # embedding_model.load_state_dict(torch.load('data/parameter_models/embedding_modelV3.pt'))
        

    test(embedding_model, model, test_loader, criterion, args.output_embedding_model_shape)





