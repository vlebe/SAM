import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import torchaudio
from torchaudio.transforms import MFCC


resolution = (3,256,256)
number_of_frames = 4
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def custom_collate(batch):
    labels, txt, mfcc_list, frame_sequence = zip(*batch)
    # Pad each sequence individually and store them in a list
    padded_mfcc_list = pad_sequence([mfcc for mfcc in mfcc_list], batch_first=True, padding_value=0)

    if len(padded_mfcc_list.shape) == 2 :
        desired_length = 3960
        padding_needed = max(0, desired_length - padded_mfcc_list.size(1))

        padded_mfcc_list = torch.nn.functional.pad(padded_mfcc_list, (0, padding_needed), value=0)

    return torch.tensor(labels), txt, padded_mfcc_list, frame_sequence

def train_test_split(dataset, test_size=0.10, val_size=0.15):
    val_size = int(len(dataset) * val_size)
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

class Dataset(Dataset):
    def __init__(self, labels_file_path, frame_dir, audio_dir, txt_data_file_path, img_shape, mlp_audio= True, preprocess = None):
        self.labels = pd.read_csv(labels_file_path)
        self.frame_dir = frame_dir
        self.audio_dir = audio_dir
        self.img_shape = img_shape
        self.txt_data = pd.read_csv(txt_data_file_path)['text']
        if preprocess is None :
            self.preprocess = transforms.Compose([
                transforms.Pad(padding=256).to(device=device),
                transforms.Resize(256).to(device=device),
                transforms.CenterCrop(256).to(device=device),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device=device),
            ])
        else :
            self.preprocess = preprocess
        self.mlp_audio = mlp_audio

    def __len__(self) :
        return len(self.labels)

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

        transform = MFCC(
            sample_rate=sampling_rate,
            n_mfcc=20,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = transform(audio).squeeze(0).T

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
    
if __name__ == "__main__" :
    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', resolution)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    for batch in dataloader:
        print(len(batch))
        break
    # print(dataset.__getitem__(0))