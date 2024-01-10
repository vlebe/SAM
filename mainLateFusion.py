import torch 
from model_video import VideoModelLateFusion1, VideoEmbedding
from model_audio import AudioMLPModel1, AudioRNNModel
from model_text import TextModel, DistilCamembertEmbedding
from dataset import Dataset, custom_collate_Dataset, train_test_split
from torch.utils.data import DataLoader
import argparse
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np



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



def train(model, dataloader, criterion, optimizer, embedding_model_video, embedding_model_text, output_embedding_model_shape):
    model.train()
    model.video_model.train()
    model.audio_model.train()
    model.text_model.train()
    embedding_model_text.train()
    embedding_model_video.train()
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

def evaluate(model, embedding_model_video, embedding_model_text, dataloader, criterion, metrics):
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--learning_rate_default', type=float, default=0.0001, help='learning rate for other layers')
    parser.add_argument('--hidden_size_video', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers_video', type=int, default=1, help='number of layers')
    parser.add_argument('--max_iter', type=int, default=20, help='maximum number of iteration, witout any improvement on the train loss with tol equal to 1e-3')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers_training', type=int, default=10, help='number of workers, corresponding to number of CPU cores that want to be used for training. 6 is recommended if available.')
    parser.add_argument('--num_workers_evaluating', type=int, default=10, help='number of workers, corresponding to number of CPU cores that want to be used for testing. 1 is recommended if available.')
    parser.add_argument('--input_size_audio_mlp', type=int, default=3960, help='input size of the audio model')
    parser.add_argument('--input_size_audio_rnn', type=int, default=20, help='input size of the audio model')
    parser.add_argument('--input_size_text', type=int, default=768, help='input size of the audio model')
    parser.add_argument('--mlp_audio', action='store_true', default=False, help='use MLP audio model instead of RNN')
    args = parser.parse_args()

    resolution = (3, 224, 224)
    output_embedding_model_shape = (2048, 1, 1)
    input_size_video = output_embedding_model_shape[0] * output_embedding_model_shape[1] * output_embedding_model_shape[2]
    embedding_model_video = VideoEmbedding().to(device)
    embedding_model_text = DistilCamembertEmbedding().to(device)
    tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=False)
    model_video = VideoModelLateFusion1(input_size_video, args.hidden_size_video, args.num_layers_video, args.num_classes).to(device)
    # model_audio = AudioMLPModel1(args.input_size_audio_mlp, args.num_classes).to(device)
    model_audio = AudioRNNModel(input_size=args.input_size_audio_rnn, num_classes=args.num_classes).to(device)
    model_text = TextModel(args.input_size_text, args.num_classes).to(device)
    model_video.load_state_dict(torch.load('data/ModelLateFusion/Models/model_video.pt'))
    model_audio.load_state_dict(torch.load('data/ModelLateFusion/Models/model_audio.pt'))   
    model_text.load_state_dict(torch.load('data/ModelLateFusion/Models/model_text.pt'))
    model = LateFusionModel(model_video, model_audio, model_text).to(device)
    
    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', args.resolution, args.output_embedding_model_shape, mlp_audio=args.mlp_audio)
    train_dataset, validation_dataset, test_dataset = train_test_split(dataset, test_size=0.10, val_size=0.15)
    workers = True
    if (device == torch.device('cuda') or device == torch.device('mps')) and workers:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_training, pin_memory=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=args.num_workers_evaluating, pin_memory=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset, shuffle=True, num_workers=args.num_workers_evaluating, pin_memory=True, collate_fn=custom_collate_Dataset)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_Dataset)
        validation_loader = DataLoader(validation_dataset, shuffle=True, collate_fn=custom_collate_Dataset)
        test_loader = DataLoader(test_dataset, shuffle=True, collate_fn=custom_collate_Dataset)
    
    weights = torch.tensor([0.1, 0.9]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    valid_losses = []
    max_iter = args.max_iter
    non_valid_iteration = 0
    models_parameters = []
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, embedding_model_video, embedding_model_text, args.output_embedding_model_shape)
        valid_loss, valid_scores = evaluate(model,embedding_model_video,embedding_model_text,validation_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score])
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
            break
        else:
            print(f"Epoch {epoch} - Training Loss: {train_loss} - Validation Loss: {valid_loss} - Validation Scores (F1 score, accuracy_score, precision score, recall score): {valid_scores}")
            valid_losses.append(valid_loss)
            train_losses.append(train_loss)

    pocket_model = LateFusionModel(model_video, model_audio, model_text).to(device)
    pocket_model.load_state_dict(models_parameters[-1])
    torch.save(pocket_model.state_dict(), 'model_late_fusion_PocketAlgo.pt')
    torch.save(model.state_dict(), 'model_late_fusionEarlyStopping.pt')
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("data/late_fusion_model_loss.png")
    _, test_scores_pocket = evaluate(pocket_model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
    _, test_scores = evaluate(model, embedding_model_video, embedding_model_text, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score])
    print(f"Test Scores (F1 score, accuracy_score, precision score, recall score): {test_scores}")
    print(f"Test Scores Pocket (F1 score, accuracy_score, precision score, recall score): {test_scores_pocket}")


