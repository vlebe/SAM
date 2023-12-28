from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
import librosa
import numpy as np

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

        y, sr = librosa.load(audio_path, sr=None)

        # Compute the mel spectrogram
        n_fft = 1024  # Length of the FFT window
        hop_length = 256  # Hop length between frames
        n_mels = 128  # Number of mel frequency bins
        fmin = 20  # Minimum frequency for mel spectrogram
        fmax = sr / 2  # Maximum frequency for mel spectrogram (Nyquist frequency)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)

        # Convert to decibels (log scale)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        resized_spectrogram = np.resize(log_mel_spectrogram, (96, 64))

        return int(label), torch.tensor(resized_spectrogram).unsqueeze(0)

if __name__ == "__main__":
    dataset = AudioDataset('data/audio/samples/', 'labels.csv')
    label, mfcc = dataset.__getitem__(9)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for labels, mfcc in dataloader:
        print(mfcc.shape)    
        print(labels.shape)
        break