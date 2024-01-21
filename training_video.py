
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
        sequence = torch.zeros((1, 4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2]), dtype=torch.float32).to(device)
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
            sequence = torch.zeros((1, 4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2]), dtype=torch.float32).to(device)
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

    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--learning_rate_default', type=float, default=0.0001, help='learning rate for other layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--max_iter', type=int, default=10, help='maximum number of iteration, witout any improvement on the train loss with tol equal to 1e-3')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing.')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model or not')
    parser.add_argument('--load_model', action='store_true', default=False, help='load model or not')
    parser.add_argument('--save_model_path', type=str, default='data/ModelVideo/Models/', help='path to save model')
    args = parser.parse_args()
    input_size = args.output_embedding_model_shape[0] * args.output_embedding_model_shape[1] * args.output_embedding_model_shape[2]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
    embedding_model = VideoEmbedding().to(device)
    weights = torch.tensor([0.2, 0.8]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    metrics = [accuracy_score, f1_score, precision_score, recall_score]

    dataset = VideoDataset('labels.csv', 'data/video/dataset_frame/', args.resolution, args.output_embedding_model_shape)
    train_dataset, validation_dataset, test_dataset = train_test_split(dataset, test_size=0.10, val_size=0.15)
    workers = True
    if (device == torch.device('cuda') or device == torch.device('mps')) and workers:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=True)
    
    # regarder la proportion de 0 et de 1 dans le train, valid et test loader
    with torch.no_grad():
        total_label_1_train = 0
        total_label_0_train = 0
        for labels, _ in tqdm(train_loader):
            total_label_1_train += labels.sum()
            total_label_0_train += (1-labels).sum()
        print(total_label_1_train, total_label_0_train)
        total_label_0_valid = 0
        total_label_1_valid = 0
        for labels, _ in tqdm(validation_loader):
            total_label_1_valid += labels.sum()
            total_label_0_valid += (1-labels).sum()
        print(total_label_1_valid, total_label_0_valid)
        total_label_0_test = 0
        total_label_1_test = 0
        for labels, _ in tqdm(test_loader):
            total_label_1_test += labels.sum()
            total_label_0_test += (1-labels).sum()
        print(total_label_1_test, total_label_0_test)
       
    if args.save_model:
        print("Start training...")
        valid_losses = []
        train_losses = []
        non_valid_iteration = 0
        models_parameters = []
        for epoch in range(args.num_epochs):
            train_loss = train(embedding_model, model, train_loader, optimizer, criterion, args.output_embedding_model_shape)
            valid_loss, val_scores = validation(embedding_model, model, validation_loader, criterion, args.output_embedding_model_shape, metrics)
            train_losses.append(train_loss)
            if epoch == 0:
                pass
            elif valid_loss < min(valid_losses):
                non_valid_iteration = 0
                models_parameters.append(model.state_dict())
            else:
                non_valid_iteration += 1
            valid_losses.append(valid_loss)
            if non_valid_iteration == args.max_iter:
                print(f"Early stopping at epoch {epoch+1} : train loss {train_loss} valid loss {valid_loss}")
                break
            else:
                print(f"Epoch {epoch+1} : train loss {train_loss} valid loss {valid_loss}")
                print(f"Validation scores : {val_scores}")

        if len(models_parameters) != 0:
            pocket_model = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
            pocket_model.load_state_dict(models_parameters[-1])
            torch.save(pocket_model.state_dict(), 'data/ModelVideo/Models/model1VFV0EarlyStoppingPocketAlgo.pt')
        torch.save(model.state_dict(), 'data/ModelVideo/Models/model1VFV0EarlyStopping.pt')
        plt.plot(train_losses, label="Training Loss")
        plt.plot(valid_losses, label="Validation Loss")
        plt.legend()
        plt.savefig("data/ModelVideo/Graphs/video_model_loss_model1VFV0.png")
        test(embedding_model, model, test_loader, criterion, args.output_embedding_model_shape)
        if len(models_parameters) != 0:
            test(embedding_model, pocket_model, test_loader, criterion, args.output_embedding_model_shape)

    if args.load_model:
        print("Loading model...")
        model.load_state_dict(torch.load('data/ModelVideo/Models/model1VFV0EarlyStopping.pt'))
        pocket_model = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
        pocket_model.load_state_dict(torch.load('data/ModelVideo/Models/model1VFV0EarlyStoppingPocketAlgo.pt'))
        test(embedding_model, model, test_loader, criterion, args.output_embedding_model_shape)
        test(embedding_model, pocket_model, test_loader, criterion, args.output_embedding_model_shape)





