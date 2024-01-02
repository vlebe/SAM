
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
from video_model import VideoModelLateFusion1, VideoModelLateFusion2, VideoEmbedding
from audio_model import AudioMLPModel1, AudioMLPModel2, AudioMLPModel3
from torch.utils.data import Dataset, DataLoader
from dataset import custom_collate, train_test_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class LateFusionModel(nn.Module):
    def __init__(self, video_model, audio_model, weight_video, weight_audio):
        super(LateFusionModel, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x_video, x_audio):
        x_video = self.video_model(x_video)
        x_audio = self.audio_model(x_audio)
        label_predicted = torch.argmax(self.Softmax(x_video * self.weight_video + x_audio * self.weight_audio), dim=1)
        # label_predicted = torch.argmax(weight_video * self.Softmax(x_video) + weight_audio * self.Softmax(x_audio), dim=1)
        return label_predicted


def train(embedding_model_video, model_video, model_audio, train_loader, optimizer_video, optimizer_audio, criterion_video, criterion_audio, output_embedding_model_shape, stop_training_video = False, stop_training_audio = False):

    print("Training...")
    global device
    device = torch.device('mps')
    embedding_model_video.train()
    model_video.train()
    model_audio.train()
    model_audio.to(device)
    model_video.to(device)
    embedding_model_video.to(device)
    running_loss_video = 0.0
    running_loss_audio = 0.0
    for i, (labels, txt, mfcc, frames) in tqdm(enumerate(train_loader)):
        if not(stop_training_video):
            sequence = torch.zeros((output_embedding_model_shape[0],4, output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]))
            if torch.backends.mps.is_available():
                labels = labels.type(torch.LongTensor)
                labels = labels.to('mps')
                frames = frames.to('mps')
                sequence = sequence.to('mps')
            optimizer_video.zero_grad()
            for j in range(frames.shape[1]):
                sequence[:,j, :] = embedding_model_video(frames[:, j, :, :, :])
            outputs_video = model_video(sequence)
            loss_video = criterion_video(outputs_video, labels)
            loss_video.backward()
            optimizer_video.step()
            running_loss_video += loss_video.item()
        else :
            print("Video model is not trained")
        if not(stop_training_audio):
            if torch.backends.mps.is_available():
                labels = labels.type(torch.LongTensor)
                labels = labels.to('mps')
                mfcc = mfcc.to('mps')
            optimizer_audio.zero_grad()
            outputs_audio = model_audio(mfcc)
            loss_audio = criterion_audio(outputs_audio, labels)
            loss_audio.backward()
            optimizer_audio.step()
            running_loss_audio += loss_audio.item()
        else :
            print("Audio model is not trained")
    return running_loss_video / len(train_loader), running_loss_audio / len(train_loader)

def validation(embedding_model_video, model_video, model_audio, validation_loader, optimizer_video, optimizer_audio, criterion_video, criterion_audio, output_embedding_model_shape, stop_training_video, stop_training_audio):
    print("Validation...")
    embedding_model_video.eval()
    model_video.eval()
    model_audio.eval()
    embedding_model_video.to('cpu')
    model_video.to('cpu')
    model_audio.to('cpu')

    running_loss_video = 0.0
    running_loss_audio = 0.0
    for i, (labels, txt, mfcc, frames) in tqdm(enumerate(validation_loader)):
        if not(stop_training_video):
            sequence = torch.zeros((1, 4, output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]), dtype=torch.float32)
            labels = labels.type(torch.LongTensor)
            labels = labels.to('cpu')
            frames = frames.to('cpu')
            sequence = sequence.to('cpu')
            for j in range(frames.shape[1]):
                sequence[:, j, :] = embedding_model_video(frames[:, j, :, :, :])
            outputs_video = model_video(sequence)
            loss_video = criterion_video(outputs_video, labels)
            running_loss_video += loss_video.item()
        else :
            print("Video model is not trained")
        if not(stop_training_audio):
            if torch.backends.mps.is_available():
                labels = labels.type(torch.LongTensor)
                labels = labels.to('mps')
                mfcc = mfcc.to('mps')
            outputs_audio = model_audio(mfcc)
            loss_audio = criterion_audio(outputs_audio, labels)
            running_loss_audio += loss_audio.item()
        else :
            print("Audio model is not trained")
    return running_loss_video / len(validation_loader), running_loss_audio / len(validation_loader)



def test(embedding_model_video, model_video, model_audio, test_loader, criterion_video, criterion_audio, output_embedding_model_shape):
    global device
    embedding_model_video.eval()
    model_video.eval()
    model_audio.eval()
    embedding_model_video.to(device)
    model_video.to(device)
    model_audio.to(device)

    total_predictions_video = torch.tensor([], dtype=torch.float32, device = device)
    total_labels_video = torch.tensor([], dtype=torch.float32, device = device)
    total_predictions_audio = torch.tensor([], dtype=torch.float32, device = device)
    total_labels_audio = torch.tensor([], dtype=torch.float32, device = device)

    running_loss_video = 0.0
    running_loss_audio = 0.0

    with torch.no_grad():
        for i, (labels, txt, mfcc, frames) in tqdm(enumerate(test_loader)):
            sequence = torch.zeros((1, 4, output_embedding_model_shape[1] * output_embedding_model_shape[2] * output_embedding_model_shape[3]), dtype=torch.float32)

            
            if torch.backends.mps.is_available():
                labels = labels.type(torch.LongTensor)
                labels = labels.to('mps')
                frames = frames.to('mps')
                sequence = sequence.to('mps')
                mfcc = mfcc.to('mps')

            for j in range(frames.shape[1]):
                sequence[:, j, :] = embedding_model_video(frames[:, j, :, :, :])

            outputs_video = model_video(sequence)
            outputs_audio = model_audio(mfcc)
            loss_video = criterion_video(outputs_video, labels)
            loss_audio = criterion_audio(outputs_audio, labels)
            running_loss_video += loss_video.item()
            running_loss_audio += loss_audio.item()

            # Concatenate predictions and labels for later metric calculation
            total_predictions_video = torch.cat((total_predictions_video, outputs_video), dim=0)
            total_labels_video = torch.cat((total_labels_video, labels), dim=0)
            total_predictions_audio = torch.cat((total_predictions_audio, outputs_audio), dim=0)
            total_labels_audio = torch.cat((total_labels_audio, labels), dim=0)


    average_loss_video = running_loss_video / len(test_loader)
    average_loss_audio = running_loss_audio / len(test_loader)
    thresholded_predictions_video = torch.argmax(total_predictions_video, dim=1)
    thresholded_predictions_audio = torch.argmax(total_predictions_audio, dim=1)
    # Calculate overall precision, recall, and accuracy
    overall_tp_video = torch.sum((thresholded_predictions_video == 1) & (total_labels_video == 1)).item()
    overall_fp_video = torch.sum((thresholded_predictions_video == 1) & (total_labels_video == 0)).item()
    overall_tn_video = torch.sum((thresholded_predictions_video == 0) & (total_labels_video == 0)).item()
    overall_fn_video = torch.sum((thresholded_predictions_video == 0) & (total_labels_video == 1)).item()

    overall_tp_audio = torch.sum((thresholded_predictions_audio == 1) & (total_labels_audio == 1)).item()
    overall_fp_audio = torch.sum((thresholded_predictions_audio == 1) & (total_labels_audio == 0)).item()
    overall_tn_audio = torch.sum((thresholded_predictions_audio == 0) & (total_labels_audio == 0)).item()
    overall_fn_audio = torch.sum((thresholded_predictions_audio == 0) & (total_labels_audio == 1)).item()

    overall_precision_video = overall_tp_video / max(overall_tp_video + overall_fp_video, 1e-10)
    overall_recall_video = overall_tp_video / max(overall_tp_video + overall_fn_video, 1e-10)
    overall_accuracy_video = (overall_tp_video + overall_tn_video) / max(overall_tp_video + overall_tn_video + overall_fp_video + overall_fn_video, 1e-10)
    overall_f1_score_video = 2 * overall_precision_video * overall_recall_video / max(overall_precision_video + overall_recall_video, 1e-10)

    overall_precision_audio = overall_tp_audio / max(overall_tp_audio + overall_fp_audio, 1e-10)
    overall_recall_audio = overall_tp_audio / max(overall_tp_audio + overall_fn_audio, 1e-10)
    overall_accuracy_audio = (overall_tp_audio + overall_tn_audio) / max(overall_tp_audio + overall_tn_audio + overall_fp_audio + overall_fn_audio, 1e-10)
    overall_f1_score_audio = 2 * overall_precision_audio * overall_recall_audio / max(overall_precision_audio + overall_recall_audio, 1e-10)

    print(f'Overall Metrics Video - Loss: {average_loss_video:.4f}, Precision: {overall_precision_video:.4f}, Recall: {overall_recall_video:.4f}, Accuracy: {overall_accuracy_video:.4f} F1 Score: {overall_f1_score_video:.4f}')
    print(f'Overall Metrics Audio - Loss: {average_loss_audio:.4f}, Precision: {overall_precision_audio:.4f}, Recall: {overall_recall_audio:.4f}, Accuracy: {overall_accuracy_audio:.4f} F1 Score: {overall_f1_score_audio:.4f}')
    return average_loss_video, average_loss_audio


def evaluate_LateFusion_Model(model, test_loader, metrics):
    model.eval()
    device = torch.device('mps')
    model.to(device)
    total_loss = 0
    scores = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for labels, _, mfcc, frames in tqdm(test_loader):
            labels = labels.type(torch.LongTensor)
            labels, mfcc = labels.to(device), mfcc.to(device)
            outputs = model(frames, mfcc)
            for i, metric in enumerate(metrics):
                scores[i] += metric(labels.cpu(), outputs.cpu().argmax(dim=1))
            scores = np.array(scores)
    return scores / len(test_loader)


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
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
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

    embedding_model_video = VideoEmbedding(pretrained_model)
    model_video = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes)
    # model_video = VideoModelLateFusion2(input_size, args.hidden_size, args.num_layers, args.num_classes)
    embedding_model_video.to(device)
    model_video.to(device)

    model_audio = AudioMLPModel1()
    model_audio.to(device)
    # model_audio = AudioMLPModel2()
    # model_audio = AudioMLPModel3()

    optimizer_video = optim.Adam(model_video.parameters(), lr=args.learning_rate)
    optimizer_audio = torch.optim.Adam(model_audio.parameters(), lr=0.0001)
    criterion_video = nn.CrossEntropyLoss().to(device)
    criterion_audio = nn.CrossEntropyLoss().to(device)
    dataset = Dataset(labels_file_path="labels.csv", frame_dir="data/video/dataset_frame/", audio_dir="data/audio/samples/", txt_data_file_path="data.csv", img_shape=(3,256,256), mlp_audio=True)
    train_dataset, val_dataset, test_dataset = train_test_split(dataset)
    

    if device == torch.device('cuda') or device == torch.device('mps'):
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate, num_workers=10, pin_memory=True)
        validation_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate, num_workers=10, pin_memory=True), 
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate, num_workers=10, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
        validation_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate), 
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    print("Start training...")
    valid_losses_video = []
    valid_losses_audio = []
    non_valid_iteration_video = 0
    non_valid_iteration_audio = 0
    max_iter = 5
    stop_training_video = False
    stop_training_audio = False
    for epoch in range(args.num_epochs):
        train_loss_video, train_loss_audio = train(embedding_model_video, model_video, model_audio, train_loader, optimizer_video, optimizer_audio, criterion_video, criterion_audio, args.output_embedding_model_shape, stop_training_video, stop_training_audio)
        valid_loss_video, valid_loss_audio = validation(embedding_model_video, model_video, model_audio, validation_loader, optimizer_video, optimizer_audio, criterion_video, criterion_audio, args.output_embedding_model_shape, stop_training_video, stop_training_audio)
        if epoch == 0:
            valid_losses_video.append(valid_loss_video)
            valid_losses_audio.append(valid_loss_audio)
        if valid_loss_video < min(valid_losses_video):
            valid_losses_video.append(valid_loss_video)
            non_valid_iteration_video = 0
        else:
            non_valid_iteration_video += 1
        if valid_loss_audio < min(valid_losses_audio):
            valid_losses_audio.append(valid_loss_audio)
            non_valid_iteration_audio = 0
        else:
            non_valid_iteration_audio += 1
        if non_valid_iteration_video == max_iter:
            print(f"Early stopping Video at epoch {epoch+1} : train loss {train_loss_video} valid loss {valid_loss_video}")
            stop_training_video = True
        if non_valid_iteration_audio == max_iter:
            print(f"Early stopping Audio at epoch {epoch+1} : train loss {train_loss_audio} valid loss {valid_loss_audio}")
            stop_training_audio = True
        if stop_training_video and stop_training_audio:
            break
        else:
            if not(stop_training_video):
                print(f"Epoch {epoch+1} : train loss {train_loss_video} valid loss {valid_loss_video}")
            if not(stop_training_audio):
                print(f"Epoch {epoch+1} : train loss {train_loss_audio} valid loss {valid_loss_audio}")


    torch.save(embedding_model_video.state_dict(), 'data/parameter_models/embedding_modelV4.pt')
    torch.save(model_video.state_dict(), 'data/parameter_models/modelV4.pt')

    # model_video = VideoModelLateFusion1(input_size, args.hidden_size, args.num_layers, args.num_classes).to(device)
    # model_video.load_state_dict(torch.load('data/parameter_models/modelV4.pt'))
    # embedding_model_video = VideoEmbedding(pretrained_model).to(device)
    # embedding_model_video.load_state_dict(torch.load('data/parameter_models/embedding_modelV4.pt'))
    # model_audio = AudioMLPModel1().to(device)
    # model_audio.load_state_dict(torch.load('data/Model_Audio/Models/audio_model_loss_MLP2VFV1.pt'))
        

    average_loss_video, average_loss_audio = test(embedding_model_video, model_video, model_audio, test_loader, criterion_video, criterion_audio, args.output_embedding_model_shape)
    weight_video, weight_audio = 0.5, 0.5
    # weight_video, weight_audio = average_loss_video / (average_loss_video + average_loss_audio), average_loss_audio / (average_loss_video + average_loss_audio)
    model_Late_Fusion = LateFusionModel(model_video, model_audio, 0.5, 0.5)
    metrics = [f1_score, accuracy_score, precision_score, recall_score]
    test_loss, test_scores = evaluate_LateFusion_Model(model_Late_Fusion, test_loader, metrics)
    print(f"Test Loss: {test_loss}")
    print(f"Test Scores (F1 score, accuracy_score, precision score, recall score): {test_scores}")






