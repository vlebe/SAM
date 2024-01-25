import torch 
from model_video import VideoModelLateFusion1, VideoEmbedding
from model_audio import AudioMLPModel1, AudioRNNModel
from model_text import TextModel,TextModel2, DistilCamembertEmbedding
from dataset import Dataset, custom_collate_Dataset, train_test_split
from torch.utils.data import DataLoader
import argparse
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from utils import EarlyStopping


class LateFusionModel(torch.nn.Module):
    def __init__(self, video_model, audio_model, text_model,num_classes=2):
        super(LateFusionModel, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.text_model = text_model
        self.weight_video = torch.nn.Parameter(torch.rand(1,num_classes))
        self.weight_audio = torch.nn.Parameter(torch.rand(1,num_classes))
        self.weight_text = torch.nn.Parameter(torch.rand(1,num_classes))
        self.bias = torch.nn.Parameter(torch.rand(1,num_classes))
        self.num_classes = num_classes

    def forward(self, x_video, x_audio, x_text):
        with torch.no_grad():
            video = self.video_model(x_video)
            audio = self.audio_model(x_audio)
            text = self.text_model(x_text)
        video = video * self.weight_video
        audio = audio * self.weight_audio
        text = text * self.weight_text
        output = video + audio + text + self.bias
        return output

class LateFusionModel2(torch.nn.Module):
    def __init__(self, video_model, audio_model, text_model,num_classes=2):
        super(LateFusionModel, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.text_model = text_model
        self.num_classes = num_classes

    def forward(self, x_video, x_audio, x_text):
        with torch.no_grad():
            video = self.video_model(x_video)
            audio = self.audio_model(x_audio)
            text = self.text_model(x_text)
        vote = torch.zeros((len(x_video), self.num_classes)).to(device)
        for i in range(len(x_video)):
            vote[i, torch.argmax(video[i, :])] += 1
            vote[i, torch.argmax(audio[i, :])] += 1
            vote[i, torch.argmax(text[i, :])] += 1
        output = torch.argmax(vote, dim=1)
        return output


def train(model, dataloader, criterion, optimizer, embedding_model_video, embedding_model_text, output_embedding_model_shape):
    model.train()
    model.video_model.eval()
    model.audio_model.eval()
    model.text_model.eval()
    embedding_model_video.eval()
    embedding_model_text.eval()
    global device, tokenizer
    total_loss = 0
    for batch in tqdm(dataloader):
        labels, txt, mfcc, frames = batch
        if len(labels) == 1:
            continue
        labels = labels.type(torch.LongTensor).to(device)
        mfcc = mfcc.to(device)
        frames = frames.to(device)
        tokenizer_output = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
        sequences = torch.zeros((len(labels),4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2])).to(device)
        with torch.no_grad():
            for j in range(frames.shape[1]):
                sequences[:,j, :] = embedding_model_video(frames[:, j, :, :, :])
            txt = embedding_model_text(input_ids, attention_mask)
        optimizer.zero_grad()
        outputs = model(sequences, mfcc, txt)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, metrics, embedding_model_video, embedding_model_text, output_embedding_model_shape):
    model.eval()
    model.video_model.eval()
    model.audio_model.eval()
    model.text_model.eval()
    embedding_model_video.eval()
    embedding_model_text.eval()
    global device, tokenizer
    total_loss = 0
    scores = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            labels, txt, mfcc, frames = batch
            labels = labels.type(torch.LongTensor).to(device)
            mfcc = mfcc.to(device)
            frames = frames.to(device)
            sequences = torch.zeros((len(labels),4, output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2])).to(device)
            tokenizer_output = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids, attention_mask = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
            with torch.no_grad():
                for j in range(frames.shape[1]):
                    sequences[:,j, :] = embedding_model_video(frames[:, j, :, :, :])
                txt = embedding_model_text(input_ids, attention_mask)
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


if __name__ == "__main__" :
    torch.manual_seed(0)
    np.random.seed(0)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--hidden_size_video', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers_video', type=int, default=1, help='number of layers')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers_training', type=int, default=6, help='number of workers, corresponding to number of CPU cores that want to be used for training. 6 is recommended if available.')
    parser.add_argument('--num_workers_evaluating', type=int, default=6, help='number of workers, corresponding to number of CPU cores that want to be used for testing. 1 is recommended if available.')
    parser.add_argument('--input_size_audio_mlp', type=int, default=3960, help='input size of the audio model')
    parser.add_argument('--input_size_audio_rnn', type=int, default=20, help='input size of the audio model')
    parser.add_argument('--input_size_text', type=int, default=768, help='input size of the audio model')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model or not')
    parser.add_argument('--load_model', action='store_true', default=False, help='load model or not')
    parser.add_argument('--mlp_audio', action='store_true', default=False, help='use MLP audio model instead of RNN')
    args = parser.parse_args()



    input_size_video = args.output_embedding_model_shape[0] * args.output_embedding_model_shape[1] * args.output_embedding_model_shape[2]
    embedding_model_video = VideoEmbedding().to(device)
    embedding_model_text = DistilCamembertEmbedding().to(device)
    tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=False)
    model_video = VideoModelLateFusion1(input_size_video, args.hidden_size_video, args.num_layers_video, args.num_classes)
    # model_audio = AudioMLPModel1(args.input_size_audio_mlp, args.num_classes).to(device)
    model_audio = AudioRNNModel(input_size=args.input_size_audio_rnn, num_classes=args.num_classes)
    model_text = TextModel2(args.input_size_text, args.num_classes)
    model_video.load_state_dict(torch.load('data/ModelLateFusion/Models/model_video_balanced.pt', map_location=torch.device('cpu')))
    model_audio.load_state_dict(torch.load('data/ModelLateFusion/Models/model_audio_balanced.pt', map_location=torch.device('cpu')))   
    model_text.load_state_dict(torch.load('data/ModelLateFusion/Models/model_text_balanced.pt', map_location=torch.device('cpu')))
    model_video = model_video.to(device)
    model_audio = model_audio.to(device)
    model_text = model_text.to(device)
    model = LateFusionModel(model_video, model_audio, model_text).to(device)
    
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
    workers = True
    if (device == torch.device('cuda') or device == torch.device('mps')) and workers:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_training, pin_memory=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_evaluating, pin_memory=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers_evaluating, pin_memory=True, collate_fn=custom_collate_Dataset)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)
    
    weights = torch.tensor([1.0, 1.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(path='data/ModelLateFusion/Models/ModelLateVFV0BalancedEarlyStopping.pt')


    if args.save_model:
        train_losses = []
        valid_losses = []
        for epoch in range(args.num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, embedding_model_video, embedding_model_text, args.output_embedding_model_shape)
            valid_loss, valid_scores = evaluate(model, validation_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], embedding_model_video, embedding_model_text, args.output_embedding_model_shape)
            early_stopping(valid_loss, model)
            valid_losses.append(valid_loss)
            train_losses.append(train_loss)
            print("Epoch {} : Train Loss = {}, Val Loss = {}, Val Scores = {}".format(epoch, train_loss, valid_loss, valid_scores))

            if early_stopping.early_stop:
                print("Early stopping")
                break

        torch.save(model.state_dict(), 'data/ModelLateFusion/Models/ModelLateVFV0Balanced.pt')
        _, test_scores = evaluate(model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], embedding_model_video, embedding_model_text, args.output_embedding_model_shape)
        print(f"Test Scores (F1 score, accuracy_score, precision score, recall score): {test_scores}")
        plt.plot(train_losses, label="Training Loss")
        plt.plot(valid_losses, label="Validation Loss")
        plt.legend()
        plt.savefig("data/ModelLateFusion/Graphs/late_fusion_model_loss_VFVOBalanced.png")
    
    if args.load_model:
        model.load_state_dict(torch.load('data/ModelLateFusion/Models/ModelLateVFV0Balanced.pt'))
        pocket_model = LateFusionModel(model_video, model_audio, model_text).to(device)
        pocket_model.load_state_dict(torch.load('data/ModelLateFusion/Models/ModelLateVFV0BalancedEarlyStopping.pt'))
        _, test_scores = evaluate(model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], embedding_model_video, embedding_model_text, args.output_embedding_model_shape)
        _, test_scores_pocket = evaluate(pocket_model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], embedding_model_video, embedding_model_text, args.output_embedding_model_shape)
        print(f"Test Scores (F1 score, accuracy_score, precision score, recall score): {test_scores}")
        print(f"Test Scores Pocket (F1 score, accuracy_score, precision score, recall score): {test_scores_pocket}")


