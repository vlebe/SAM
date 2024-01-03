from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
import librosa
from vggish_input import wavfile_to_examples
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def custom_collate(batch):
    # Extract MFCCs from each tuple in the batch
    log_mels_batch = [sample[1] for sample in batch]

    # Get the maximum length along the first dimension in the batch
    max_length = max(mfcc.size(0) for mfcc in log_mels_batch)

    # Pad each MFCC in the batch with zeros along the first dimension
    padded_batch = [
        F.pad(mfcc, (0, 0, 0, max_length - mfcc.size(0), 0, 0, 0, 0), value=0)
        for mfcc in log_mel
    ]

    # Stack the padded MFCCs to create the batch
    padded_batch = torch.stack(padded_batch, dim=0)

    # Extract labels from each tuple in the batch
    labels_batch = [sample[0] for sample in batch]

    return labels_batch, padded_batch


def train_test_split(dataset, test_size=0.10, val_size=0.15):
    val_size = int(len(dataset) * val_size)
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_file):
        self.audio_dir = audio_dir
        self.labels = pd.read_csv(label_file)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels["turn_after"][index]

        speaker = self.labels["speaker"][index]
        if type(speaker) != str:
            speaker = "NA"
        record = self.labels["dyad"][index]
        audio_path = self.audio_dir + f"{record}/{record}_{speaker}_{index}.wav"
        log_mel = wavfile_to_examples(audio_path)

        return int(label), log_mel

if __name__ == "__main__":
    dataset = AudioDataset('data/audio/samples/', 'labels.csv')
    label, log_mel = dataset.__getitem__(9)
    print(log_mel.shape)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    for labels, mfcc in dataloader:
        print(mfcc.shape)    
        print(labels.shape)
        break