import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision 
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import torchaudio
from torchaudio.transforms import MFCC
from tqdm import tqdm
from model_audio import AudioMLPModel1, AudioRNNModel
import torch


resolution = (3, 224, 224)
number_of_frames = 4


def custom_collate_Dataset(batch):
    labels, txt, mfcc_list, frame_sequence = zip(*batch)
    padded_mfcc_list = pad_sequence([mfcc for mfcc in mfcc_list], batch_first=True, padding_value=0)

    if len(padded_mfcc_list.shape) == 2 :
        desired_length = 3960
        padding_needed = max(0, desired_length - padded_mfcc_list.size(1))

        padded_mfcc_list = torch.nn.functional.pad(padded_mfcc_list, (0, padding_needed), value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    txt = [txt[i] for i in range(len(txt)) if len(txt[i]) > 0]
    frame_sequence = torch.stack(frame_sequence)
    return labels, txt, padded_mfcc_list, frame_sequence

def custom_collate_AudioDataset(batch):
    labels, mfcc_list = zip(*batch)
    # Pad each sequence individually and store them in a list
    padded_mfcc_list = pad_sequence([mfcc for mfcc in mfcc_list], batch_first=True, padding_value=0)
    if len(padded_mfcc_list.shape) == 2 :
        desired_length = 3960
        padding_needed = max(0, desired_length - padded_mfcc_list.size(1))

        padded_mfcc_list = torch.nn.functional.pad(padded_mfcc_list, (0, padding_needed), value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return labels, padded_mfcc_list


def train_test_split(dataset, test_size=0.10, val_size=0.15):
    val_size = int(len(dataset) * val_size)
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

class Dataset(Dataset):
    def __init__(self, labels_file_path, frame_dir, audio_dir, txt_data_file_path, img_shape, output_embedding_model_shape, mlp_audio= True):
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.labels = pd.read_csv(labels_file_path)
        self.frame_dir = frame_dir
        self.audio_dir = audio_dir
        self.img_shape = img_shape
        self.output_embedding_model_shape = output_embedding_model_shape
        self.txt_data = pd.read_csv(txt_data_file_path)['text']
        self.preprocess = weights.transforms()
        self.mlp_audio = mlp_audio

    def __len__(self) :
        return len(self.labels)
    
    def transform(self, sampling_rate):
        return MFCC(
            sample_rate=sampling_rate,
            n_mfcc=20,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

    def __getitem__(self, index):
        if type(index) == torch.Tensor :
            index = index.item()
        label = self.labels["turn_after"][index]
        speaker = self.labels["speaker"][index]
        if type(speaker) != str:
            speaker = "NA"
        record = self.labels["dyad"][index]
        audio_path = self.audio_dir + f"{record}/{record}_{speaker}_{index}.wav"
        audio, sampling_rate = torchaudio.load(audio_path, normalize=True)
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0, keepdim=True)
        mfcc = self.transform(sampling_rate)(audio).squeeze(0).T
        if self.mlp_audio :
            mfcc = torch.nn.Flatten()(mfcc.unsqueeze(0))
        txt = self.txt_data[index]
        frame_dir = self.frame_dir + f"IPU_{index}/"
        img_list_path = os.listdir(frame_dir)
        frame_sequence = np.zeros((4, self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        for i, image_path in enumerate(img_list_path):
            img = Image.open(frame_dir + image_path) 
            img = transforms.Pad(padding=256)(img)
            img = self.preprocess(img)
            frame_sequence[i] = img
        frame_sequence = torch.tensor(frame_sequence, dtype=torch.float32)

        return int(label), txt, mfcc.squeeze(0), frame_sequence
    

class AudioDataset(Dataset):
    def __init__(self, labels_file_path, audio_dir, mlp_audio= True):
        self.labels = pd.read_csv(labels_file_path)
        self.audio_dir = audio_dir
        self.mlp_audio = mlp_audio

    def __len__(self) :
        return len(self.labels)
    
    def transform(self, sampling_rate):
        return MFCC(
            sample_rate=sampling_rate,
            n_mfcc=20,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

    def __getitem__(self, index):
        if type(index) == torch.Tensor :
            index = index.item()
        label = self.labels["turn_after"][index]
        speaker = self.labels["speaker"][index]
        if type(speaker) != str:
            speaker = "NA"
        record = self.labels["dyad"][index]
        audio_path = self.audio_dir + f"{record}/{record}_{speaker}_{index}.wav"
        audio, sampling_rate = torchaudio.load(audio_path, normalize=True)
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0, keepdim=True)
        mfcc = self.transform(sampling_rate)(audio).squeeze(0).T
        if self.mlp_audio :
            mfcc = torch.nn.Flatten()(mfcc.unsqueeze(0))
        return int(label), mfcc.squeeze(0)
    

class VideoDataset(Dataset):
    global device
    def __init__(self, labels_file_path, frame_dir, img_shape, output_embedding_model_shape):
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.labels = pd.read_csv(labels_file_path)
        self.frame_dir = frame_dir
        self.img_shape = img_shape
        self.output_embedding_model_shape = output_embedding_model_shape
        self.preprocess = weights.transforms()

    def __len__(self) :
        return len(self.labels)
    
    def __getitem__(self, index):
        if type(index) == torch.Tensor :
            index = index.item()
        label = self.labels["turn_after"][index]
        frame_dir = self.frame_dir + f"IPU_{index}/"
        img_list_path = os.listdir(frame_dir)
        frame_sequence = np.zeros((4, self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        for i, image_path in enumerate(img_list_path):
            img = Image.open(frame_dir + image_path) 
            img = transforms.Pad(padding=256)(img)
            img = self.preprocess(img)
            frame_sequence[i] = img
        frame_sequence = torch.tensor(frame_sequence, dtype=torch.float32)

        return int(label), frame_sequence
    

class TextDataset(Dataset):
    def __init__(self, labels_file_path, txt_data_file_path):
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.labels = pd.read_csv(labels_file_path)
        self.txt_data = pd.read_csv(txt_data_file_path)['text']

    def __len__(self) :
        return len(self.labels)

    def __getitem__(self, index):
        if type(index) == torch.Tensor :
            index = index.item()
        label = self.labels["turn_after"][index]
        txt = self.txt_data[index]
        return int(label), txt


    
if __name__ == "__main__" :
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')


    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', resolution, (2048, 1, 1), mlp_audio=False)
    labels = dataset.labels["turn_after"].values

    #Printing the number of samples per class
    class_sample_count = dataset.labels["turn_after"].value_counts().values
    total_samples = len(dataset)
    print(class_sample_count)
    print(total_samples)

    #Doing the same but by using subset and indices
    class_0_indices = [i for i in range(len(dataset)) if labels[i] == 0]
    class_1_indices = [i for i in range(len(dataset)) if labels[i] == 1]
    #subsampling randomly class 0 to have the same number of samples as class 1
    subsampled_indices_0 = np.random.choice(class_0_indices, len(class_1_indices), replace=False)
    subsampled_indices = subsampled_indices_0.tolist() + class_1_indices

    subdataset = torch.utils.data.Subset(dataset, subsampled_indices)
    dataloader = DataLoader(subdataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=custom_collate_Dataset)
    for batch in tqdm(dataloader):
        labels, txt, mfcc_list, frame_sequence = batch
        pass
    
    dataset = AudioDataset('labels.csv', 'data/audio/samples/', mlp_audio=False)
    model_audio = AudioRNNModel().to(device)
    subdataset = torch.utils.data.Subset(dataset, subsampled_indices)
    dataloader = DataLoader(subdataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=custom_collate_Dataset)
    for batch in tqdm(dataloader):
        labels, mfcc = batch
        exemplar = mfcc.to(device)
        outputs = model_audio(exemplar)

    dataset = VideoDataset('labels.csv', 'data/video/dataset_frame/', resolution, (2048, 1, 1))
    subdataset = torch.utils.data.Subset(dataset, subsampled_indices)
    dataloader = DataLoader(subdataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=custom_collate_Dataset)
    for batch in dataloader:
        labels, frame_sequence = batch
        print(frame_sequence.size())
        exemplar = frame_sequence[:, 0, :, :, :].to(device)
        print(exemplar.size())

    dataset = TextDataset('labels.csv', 'txt_data.csv')
    subdataset = torch.utils.data.Subset(dataset, subsampled_indices)
    dataloader = DataLoader(subdataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=custom_collate_Dataset)
    for batch in dataloader:
        labels, txt = batch
        print(len(txt))