
import torch
import torch.nn as nn
import torch
import torch.optim as optim
from dataset import Dataset, custom_collate_Dataset, train_test_split
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
import torchvision 
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from model_video import VideoEmbedding
from model_text import DistilCamembertEmbedding
from transformers import AutoTokenizer


class EarlyFusionModel(nn.Module):
    def __init__(self, input_size_video, input_size_audio, input_size_txt, hidden_size_gru, num_layers_gru, num_classes):
        super(EarlyFusionModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size_video, hidden_size=hidden_size_gru, num_layers=num_layers_gru, batch_first=True)
        self.fci_audio = nn.Linear(input_size_audio, hidden_size_gru)
        self.fci_text = nn.Linear(input_size_txt, hidden_size_gru)
        self.weight_audio = nn.Parameter(torch.randn(hidden_size_gru, 1))
        self.weight_text = nn.Parameter(torch.randn(hidden_size_gru, 1))
        self.weight_video = nn.Parameter(torch.randn(hidden_size_gru, 1))
        self.fh = nn.Linear(hidden_size_gru, hidden_size_gru//4)
        self.fo = nn.Linear(hidden_size_gru//4, num_classes)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_size_gru//4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_video, x_audio, x_text):
        x_video, _ = self.gru(x_video)
        x_video = x_video[:, -1, :]
        x_audio = self.fci_audio(x_audio)
        x_text = self.fci_text(x_text)
        x = x_video * self.weight_video.T + x_audio * self.weight_audio.T + x_text * self.weight_text.T
        x = self.fh(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fo(x)
        return x



def train(embedding_model_video,embedding_model_text, model, train_loader, optimizer, criterion, output_embedding_model_shape):
    print("Training...")
    global tokenizer
    model.train()
    embedding_model_text.train()
    embedding_model_video.train()
    running_loss = 0.0
    for batch in tqdm(train_loader):
        labels, txt, mfcc, frames = batch
        if len(labels) == 1:
            continue
        frames = frames.to(device)
        mfcc = mfcc.to(device)
        sequences = torch.zeros((len(labels),4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2])).to(device)
        tokenizer_output = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
        with torch.no_grad():
            for j in range(frames.shape[1]):
                sequences[:,j, :] = embedding_model_video(frames[:, j, :, :, :])
            txt = embedding_model_text(input_ids, attention_mask).to(device)
            labels = labels.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        outputs = model(sequences, mfcc, txt)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(embedding_model_video,embedding_model_text, model, dataloader, criterion, metrics, output_embedding_model_shape):
    print("Evaluating...")
    global tokenizer
    model.eval()
    embedding_model_text.eval()
    embedding_model_video.eval()
    total_loss = 0
    scores = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            labels, txt, mfcc, frames = batch
            frames = frames.to(device)
            mfcc = mfcc.to(device)
            sequences = torch.zeros((len(labels),4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2])).to(device)
            tokenizer_output = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids, attention_mask = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
            with torch.no_grad():
                for j in range(frames.shape[1]):
                    sequences[:,j, :] = embedding_model_video(frames[:, j, :, :, :])
                txt = embedding_model_text(input_ids, attention_mask).to(device)
                labels = labels.type(torch.LongTensor).to(device)
            outputs = model(sequences, mfcc, txt)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
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
    return total_loss / len(dataloader), scores / len(dataloader)



if __name__ == '__main__':

    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--learning_rate_default', type=float, default=0.0001, help='learning rate for other layers')
    parser.add_argument('--hidden_size_gru', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers_gru', type=int, default=1, help='number of layers')
    parser.add_argument('--max_iter', type=int, default=10, help='maximum number of iteration, witout any improvement on the train loss with tol equal to 1e-3')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing. 6 is recommended if available.')
    parser.add_argument('--input_size_audio', type=int, default=3960, help='input size of the audio model')
    parser.add_argument('--input_size_text', type=int, default=768, help='input size of the audio model')
    args = parser.parse_args()
    input_size_video = args.output_embedding_model_shape[0] * args.output_embedding_model_shape[1] * args.output_embedding_model_shape[2]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = EarlyFusionModel(input_size_video, args.input_size_audio, args.input_size_text, args.hidden_size_gru, args.num_layers_gru, args.num_classes).to(device)
    embedding_model_video = VideoEmbedding().to(device)
    embedding_model_text = DistilCamembertEmbedding().to(device)
    tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=False)

    weights = torch.tensor([0.2, 0.8]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', args.resolution, args.output_embedding_model_shape)
    train_dataset, validation_dataset, test_dataset = train_test_split(dataset, test_size=0.10, val_size=0.15)
    workers = True
    if (device == torch.device('cuda') or device == torch.device('mps')) and workers:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_Dataset)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, shuffle=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset, shuffle=True, collate_fn=custom_collate_Dataset)

    print("Start training...")
    valid_losses = []
    train_losses = []
    non_valid_iteration = 0
    max_iter = 20
    models_parameters = []
    for epoch in range(args.num_epochs):
        train_loss = train(embedding_model_video,embedding_model_text, model, train_loader, optimizer, criterion, args.output_embedding_model_shape)
        valid_loss, valid_scores = evaluate(embedding_model_video,embedding_model_text, model, validation_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
        if epoch == 0:
            valid_losses.append(valid_loss)
        elif valid_loss < min(valid_losses):
            valid_losses.append(valid_loss)
            non_valid_iteration = 0
            models_parameters.append(model.state_dict())
        else:
            non_valid_iteration += 1
        if non_valid_iteration == max_iter:
            print(f"Early stopping at epoch {epoch+1}")
            pocket_model = EarlyFusionModel().to(device)
            pocket_model.load_state_dict(models_parameters[-1])
            break
        else:
            print(f"Epoch {epoch} - Training Loss: {train_loss} - Validation Loss: {valid_loss} - Validation Scores (F1 score, accuracy_score, precision score, recall score): {valid_scores}")
            valid_losses.append(valid_loss)
            train_losses.append(train_loss)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("data/ModelEarlyFusion/Graphs/earlyFusionModelLossDebog.png")

    torch.save(model.state_dict(), 'data/ModelEarlyFusion/Models/modelVdebogEarlyStopping.pt')
    torch.save(pocket_model.state_dict(), 'data/ModelEarlyFusion/Models/modelVdebogEarlyStoppingPocketAlgo.pt')
    model = EarlyFusionModel(input_size_video, args.input_size_audio, args.input_size_text, args.hidden_size_gru, args.num_layers_gru, args.num_classes).to(device)
    pocket_model = EarlyFusionModel(input_size_video, args.input_size_audio, args.input_size_text, args.hidden_size_gru, args.num_layers_gru, args.num_classes).to(device)
    # model.load_state_dict(torch.load('data/ModelEarlyFusion/Models/modelVdebogEarlyStopping.pt'))
    # pocket_model.load_state_dict(torch.load('data/ModelEarlyFusion/Models/modelVdebogEarlyStoppingPocketAlgo.pt'))
    _, test_scores = evaluate(embedding_model_video,embedding_model_text, model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
    _, pocket_test_scores = evaluate(embedding_model_video,embedding_model_text, pocket_model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
    print(f"Test Scores (F1 score, accuracy_score, precision score, recall score): {test_scores}")
    print(f"Pocket Test Scores (F1 score, accuracy_score, precision score, recall score): {pocket_test_scores}")






