
import torch
import torch.nn as nn
import torch
import torch.optim as optim
from dataset import Dataset, custom_collate_Dataset, train_test_split
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from model_video import VideoEmbedding
from model_text import DistilCamembertEmbedding
from transformers import AutoTokenizer
from utils import EarlyStopping


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
    embedding_model_text.eval()
    embedding_model_video.eval()
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
    torch.manual_seed(0)
    np.random.seed(0)

    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--hidden_size_gru', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers_gru', type=int, default=1, help='number of layers')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing. 6 is recommended if available.')
    parser.add_argument('--input_size_audio', type=int, default=3960, help='input size of the audio model')
    parser.add_argument('--input_size_text', type=int, default=768, help='input size of the audio model')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model or not')
    parser.add_argument('--load_model', action='store_true', default=True, help='load model or not')
    parser.add_argument('--mlp_audio', action='store_true', default=True, help='use MLP audio model instead of RNN')
    args = parser.parse_args()
    input_size_video = args.output_embedding_model_shape[0] * args.output_embedding_model_shape[1] * args.output_embedding_model_shape[2]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', args.resolution, args.output_embedding_model_shape, mlp_audio=args.mlp_audio)
    labels = dataset.labels["turn_after"].values
    #Doing the same but by using subset and indices
    class_0_indices = [i for i in range(len(dataset)) if labels[i] == 0]
    class_1_indices = [i for i in range(len(dataset)) if labels[i] == 1]
    #subsampling randomly class 0 to have the same number of samples as class 1
    subsampled_indices_0 = np.random.choice(class_0_indices, len(class_1_indices), replace=False)
    subsampled_indices = subsampled_indices_0.tolist() + class_1_indices
    subdataset = torch.utils.data.Subset(dataset, subsampled_indices)
    train_dataset, validation_dataset, test_dataset = train_test_split(subdataset, test_size=0.10, val_size=0.15)

    if device == torch.device('cuda') or device == torch.device('mps'):
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_Dataset)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)

    model = EarlyFusionModel(input_size_video, args.input_size_audio, args.input_size_text, args.hidden_size_gru, args.num_layers_gru, args.num_classes).to(device)
    embedding_model_video = VideoEmbedding().to(device)
    embedding_model_text = DistilCamembertEmbedding().to(device)
    tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=False)
    weights = torch.tensor([0.2, 0.8]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    early_stopping = EarlyStopping(path='data/ModelLateFusion/Models/ModelEarlyVFV0ImbalancedEarlyStopping.pt')

    if args.save_model : 
        print("Start training...")
        valid_losses = []
        train_losses = []
        for epoch in range(args.num_epochs):
            train_loss = train(embedding_model_video,embedding_model_text, model, train_loader, optimizer, criterion, args.output_embedding_model_shape)
            valid_loss, valid_scores = evaluate(embedding_model_video,embedding_model_text, model, validation_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
            early_stopping(valid_losses, model)
            valid_losses.append(valid_losses)
            train_losses.append(train_loss)
            print("Epoch {} : Train Loss = {}, Val Loss = {}, Val Scores = {}".format(epoch, train_loss, valid_loss, valid_scores))

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        plt.plot(train_losses, label="Training Loss")
        plt.plot(valid_losses, label="Validation Loss")
        plt.legend()
        plt.savefig("data/ModelEarlyFusion/Graphs/ModelEarlyVFV0Imbalanced.png")
        torch.save(model.state_dict(), 'data/ModelLateFusion/Models/ModelEarlyVFV0Imbalanced.pt')
        _, test_scores = evaluate(embedding_model_video,embedding_model_text, model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
        print(f"Test Scores F1 score : {test_scores[0]}, accuracy_score : {test_scores[1]}, precision score : {test_scores[2]}, recall score : {test_scores[3]}")
    
    if args.load_model:
        print("Loading balanced and imbalanced models...")
        balanced_model = EarlyFusionModel(input_size_video, args.input_size_audio, args.input_size_text, args.hidden_size_gru, args.num_layers_gru, args.num_classes)
        balanced_model.load_state_dict(torch.load('data/ModelEarlyFusion/Models/model_EarlyFusion_Balanced.pt', map_location=torch.device('cpu')))
        balanced_model.to(device)
        _, balanced_test_scores = evaluate(embedding_model_video,embedding_model_text, balanced_model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
        print(f"Balanced Test Scores F1 score : {balanced_test_scores[0]}, accuracy_score : {balanced_test_scores[1]}, precision score : {balanced_test_scores[2]}, recall score : {balanced_test_scores[3]}")
        imbalanced_model = EarlyFusionModel(input_size_video, args.input_size_audio, args.input_size_text, args.hidden_size_gru, args.num_layers_gru, args.num_classes)
        imbalanced_model.load_state_dict(torch.load('data/ModelEarlyFusion/Models/model_EarlyFusion_Imbalanced.pt', map_location=torch.device('cpu')))
        imbalanced_model.to(device)
        _, imbalanced_test_scores = evaluate(embedding_model_video,embedding_model_text, imbalanced_model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
        print(f"Imbalanced Test Scores F1 score : {imbalanced_test_scores[0]}, accuracy_score : {imbalanced_test_scores[1]}, precision score : {imbalanced_test_scores[2]}, recall score : {imbalanced_test_scores[3]}")

    






