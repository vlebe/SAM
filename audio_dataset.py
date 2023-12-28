from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import torchaudio
from torchaudio.transforms import MFCC

def custom_collate(batch):
    labels, mfcc_list = zip(*batch)
    # Pad each sequence individually and store them in a list
    padded_mfcc_list = pad_sequence([mfcc for mfcc in mfcc_list], batch_first=True, padding_value=0)

    if len(padded_mfcc_list.shape) == 2 :
        desired_length = 3960
        padding_needed = max(0, desired_length - padded_mfcc_list.size(1))

        padded_mfcc_list = torch.nn.functional.pad(padded_mfcc_list, (0, padding_needed), value=0)

    return torch.tensor(labels), padded_mfcc_list

def train_test_split(dataset, test_size=0.10, val_size=0.15):
    val_size = int(len(dataset) * val_size)
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_file, mlp=False):
        self.audio_dir = audio_dir
        self.labels = pd.read_csv(label_file)
        self.mlp = mlp
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
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

        if self.mlp :
            mfcc = torch.nn.Flatten()(mfcc.unsqueeze(0))

        return int(label), mfcc.squeeze(0)

if __name__ == "__main__":
    dataset = AudioDataset('data/audio/samples/', 'labels.csv', mlp=True)
    label, mfcc = dataset.__getitem__(9)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    for labels, mfcc in dataloader:
        print(mfcc.shape)    
        break