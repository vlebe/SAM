
import torch
import torch.nn as nn
import torch
import torch.optim as optim
from dataset import Dataset
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from dataset import train_test_split, VideoDataset
from torch.utils.data import DataLoader
from model_video import VideoEmbedding, VideoModelLateFusion1, VideoModelLateFusion2
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils import EarlyStopping



def train(embedding_model, model, train_loader, optimizer, criterion, output_embedding_model_shape):
    print("Training...")
    model.train()
    embedding_model.eval()
    running_loss = 0.0
    for labels, frames in tqdm(train_loader):
        labels = labels.type(torch.LongTensor).to(device)
        if labels.shape[0] != args.batch_size:
            continue
        frames = frames.to(device)
        sequences = torch.zeros((len(labels),4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2])).to(device)
        with torch.no_grad():
            for j in range(frames.shape[1]):
                sequences[:,j, :] = embedding_model(frames[:, j, :, :, :])
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validation(embedding_model, model, validation_loader, criterion, output_embedding_model_shape, metrics):
    print("Validation...")
    model.eval()
    embedding_model.eval()
    running_loss = 0.0
    scores = [0 for _ in range(len(metrics))]
    for labels, frames in tqdm(validation_loader):
        labels = labels.type(torch.LongTensor).to(device)
        frames = frames.to(device)
        sequence = torch.zeros((len(labels), 4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2]), dtype=torch.float32).to(device)
        with torch.no_grad():
            for j in range(frames.shape[1]):
                sequence[:, j, :] = embedding_model(frames[:, j, :, :, :])
        outputs = model(sequence)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        for i, metric in enumerate(metrics):
            if metric.__name__ == "accuracy_score":
                args = {'y_true': labels.cpu().numpy(), 
                    'y_pred': torch.argmax(outputs, dim=1).cpu().numpy()}
            else:
                args = {'y_true': labels.cpu().numpy(), 
                        'y_pred': torch.argmax(outputs, dim=1).cpu().numpy(),
                        'zero_division': 0.0}
            scores[i] += metric(**args)
        scores = np.array(scores) 
    return running_loss / len(validation_loader), scores/ len(validation_loader)



def test(embedding_model, model, test_loader, criterion, output_embedding_model_shape):
    print("Testing...")
    model.eval()
    embedding_model.eval()
    running_loss = 0.0
    total_predictions = torch.tensor([], dtype=torch.float32, device = device)
    total_labels = torch.tensor([], dtype=torch.float32, device = device)
    with torch.no_grad():
        for labels, frames in tqdm(test_loader):
            labels = labels.type(torch.LongTensor).to(device)
            frames = frames.to(device)
            sequence = torch.zeros((len(labels), 4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2]), dtype=torch.float32).to(device)
            for j in range(frames.shape[1]):
                sequence[:, j, :] = embedding_model(frames[:, j, :, :, :])
            outputs = model(sequence)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # Concatenate predictions and labels for later metric calculation
            total_predictions = torch.cat((total_predictions, outputs), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)
    average_loss = running_loss / len(test_loader)
    thresholded_predictions = torch.argmax(total_predictions, dim=1)
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
    torch.manual_seed(0)
    np.random.seed(0)
    
    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing.')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model or not')
    parser.add_argument('--load_model', action='store_true', default=True, help='load model or not')
    args = parser.parse_args()
    input_size = args.output_embedding_model_shape[0] * args.output_embedding_model_shape[1] * args.output_embedding_model_shape[2]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    dataset = VideoDataset('labels.csv', 'data/video/dataset_frame/', args.resolution, args.output_embedding_model_shape)
    labels = dataset.labels["turn_after"].values
    #Doing the same but by using subset and indices
    class_0_indices = [i for i in range(len(dataset)) if labels[i] == 0]
    class_1_indices = [i for i in range(len(dataset)) if labels[i] == 1]
    #subsampling randomly class 0 to have the same number of samples as class 1
    subsampled_indices_0 = np.random.choice(class_0_indices, len(class_1_indices), replace=False)
    subsampled_indices = subsampled_indices_0.tolist() + class_1_indices
    subdataset = torch.utils.data.Subset(dataset, subsampled_indices)
    train_dataset, validation_dataset, test_dataset = train_test_split(subdataset, test_size=0.10, val_size=0.15)
    workers = True
    if (device == torch.device('cuda') or device == torch.device('mps')) and workers:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        validation_loader = DataLoader(validation_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
    embedding_model = VideoEmbedding().to(device)
    weights = torch.tensor([1.0, 1.0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    metrics = [accuracy_score, f1_score, precision_score, recall_score]
    early_stopping = EarlyStopping(path='data/ModelVideo/Models/model1VFV1EarlyStoppingBalance.pt')

       
    if args.save_model:
        print("Start training...")
        valid_losses = []
        train_losses = []
        for epoch in range(args.num_epochs):
            train_loss = train(embedding_model, model, train_loader, optimizer, criterion, args.output_embedding_model_shape)
            valid_loss, val_scores = validation(embedding_model, model, validation_loader, criterion, args.output_embedding_model_shape, metrics)
            train_losses.append(train_loss)
            early_stopping(valid_loss, model)
            valid_losses.append(valid_loss)
            print("Epoch {} : Train Loss = {}, Val Loss = {}, Val Scores = {}".format(epoch, train_loss, valid_loss, val_scores))

            if early_stopping.early_stop:
                print("Early stopping")
                break

        torch.save(model.state_dict(), 'data/ModelVideo/Models/model1VFV1Balance.pt')
        plt.plot(train_losses, label="Training Loss")
        plt.plot(valid_losses, label="Validation Loss")
        plt.legend()
        plt.savefig("data/ModelVideo/Graphs/video_model_loss_model1VFV1Balance.png")
        test(embedding_model, model, test_loader, criterion, args.output_embedding_model_shape)
        pocket_model = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
        pocket_model.load_state_dict(torch.load('data/ModelVideo/Models/model1VFV1EarlyStoppingBalance.pt'))
        test(embedding_model, pocket_model, test_loader, criterion, args.output_embedding_model_shape)


    if args.load_model:
        print("Loading model...")
        model.load_state_dict(torch.load('data/ModelVideo/Models/model1VFV1Balance.pt'))
        pocket_model = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
        pocket_model.load_state_dict(torch.load('data/ModelVideo/Models/model1VFV1EarlyStoppingBalance.pt'))
        test(embedding_model, model, test_loader, criterion, args.output_embedding_model_shape)
        test(embedding_model, pocket_model, test_loader, criterion, args.output_embedding_model_shape)





