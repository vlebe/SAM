import torch 
from video_model import VideoModelLateFusion1
from audio_model import AudioMLPModel1
from text_model import TextModel
from dataset import Dataset, custom_collate_Dataset, train_test_split
from torch.utils.data import DataLoader
import argparse
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        with torch.no_grad:
            video = self.video_model(x_video)
            audio = self.audio_model(x_audio)
            text = self.text_model(x_text)
        video = video * self.weight_video
        audio = audio * self.weight_audio
        text = text * self.weight_text
        output = video + audio + text + self.bias
        return output



def train(model, dataloader, criterion, optimizer):
    model.train()
    global device
    total_loss = 0
    for batch in dataloader:
        labels, txt, mfcc, frames = batch
        labels = labels.type(torch.LongTensor).to(device)
        mfcc = mfcc.to(device)
        frames = frames.to(device)
        optimizer.zero_grad()
        outputs = model(frames, mfcc, txt)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, metrics):
    model.eval()
    global device
    total_loss = 0
    scores = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for batch in dataloader:
            labels, txt, mfcc, frames = batch
            labels = labels.type(torch.LongTensor).to(device)
            mfcc = mfcc.to(device)
            frames = frames.to(device)
            outputs = model(frames, mfcc, txt)
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

    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--learning_rate_default', type=float, default=0.0001, help='learning rate for other layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--max_iter', type=int, default=10, help='maximum number of iteration, witout any improvement on the train loss with tol equal to 1e-3')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--resolution', type=tuple, default=(3, 224, 224), help='resolution of the input images')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing. 6 is recommended if available.')
    parser.add_argument('--max_iter', type=int, default=10, help='maximum number of iteration, witout any improvement on the valid loss with tol equal to 1e-3')
    args = parser.parse_args()

    resolution = (3, 224, 224)
    output_embedding_model_shape = (2048, 1, 1)
    model_video = VideoModelLateFusion1().to(device)
    model_audio = AudioMLPModel1().to(device)
    model_text = TextModel().to(device)
    model_video.load_state_dict(torch.load('model_video.pt'))
    model_audio.load_state_dict(torch.load('model_audio.pt'))   
    model_text.load_state_dict(torch.load('model_text.pt'))
    model = LateFusionModel(model_video, model_audio, model_text).to(device)
    
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
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    valid_losses = []
    max_iter = args.max_iter
    non_valid_iteration = 0
    models_parameters = []
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, args.output_embedding_model_shape)
        valid_loss, valid_scores = evaluate(model, validation_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
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
    torch.save(pocket_model.state_dict(), 'model_late_fusion.pt')
    torch.save(model_video.state_dict(), 'model_video.pt')
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("data/late_fusion_model_loss.png")
    _, test_scores_pocket = evaluate(pocket_model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
    _, test_scores = evaluate(model, test_loader, criterion, [f1_score, accuracy_score, precision_score, recall_score], args.output_embedding_model_shape)
    print(f"Test Scores (F1 score, accuracy_score, precision score, recall score): {test_scores}")
    print(f"Test Scores Pocket (F1 score, accuracy_score, precision score, recall score): {test_scores_pocket}")


