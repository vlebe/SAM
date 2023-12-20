from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import librosa
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

def custom_collate(batch):
    labels, mfcc_list = zip(*batch)

    # Find the maximum sequence length in the batch
    max_seq_length = max([mfcc.shape[1] for mfcc in mfcc_list])

    # Pad each sequence individually and store them in a list
    padded_mfcc_list = pad_sequence([mfcc.T for mfcc in mfcc_list], batch_first=True, padding_value=0)

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
        audio, sampling_rate = librosa.load(audio_path)

        n_mfcc = 20
        hop_length = 512
        n_fft = 2048
        mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

        if self.mlp :
            print(mfcc)
            mfcc = mfcc.flatten()
            padding_needed = max(0, 100 - len(mfcc))
            pad_before = padding_needed // 2
            pad_after = padding_needed - pad_before
            mfcc = np.pad(mfcc, (pad_before, pad_after), mode='constant', constant_values=0)

        return int(label), torch.tensor(mfcc)


if __name__ == "__main__":
    dataset = AudioDataset('data/audio/samples/', 'labels.csv', mlp=True)
    print(dataset.__getitem__(0))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    for labels, mfcc in dataloader:
        print(labels)
        print(mfcc.shape)
        break