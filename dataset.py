from torch.utils.data import Dataset
import pandas as pd
from scipy.io import wavfile
import os
import numpy as np
import cv2
import torch

resolution = (70, 126, 3)
number_of_frames = 4

class Dataset(Dataset):
    def __init__(self, labels_file_path, frame_dir, audio_dir, txt_data_file_path, img_shape):
        self.labels = pd.read_csv(labels_file_path)
        self.frame_dir = frame_dir
        self.audio_dir = audio_dir
        self.img_shape = img_shape
        self.txt_data = pd.read_csv(txt_data_file_path)['text']

    def __len__(self) :
        return len(self.labels)

    def __getitem__(self, index):
        if type(index) == torch.Tensor :
            index = index.item()
        label = self.labels["turn_after"][index]

        txt = self.txt_data[index]

        speaker = self.labels["speaker"][index]
        record = self.labels["dyad"][index]
        # audio_path = self.audio_dir + f"{record}/{record}_{speaker}_{index}.wav"
        # fs, audio = wavfile.read(audio_path)

        frame_dir = self.frame_dir + f"IPU_{index}/"
        img_list_path = os.listdir(frame_dir)

        frame_sequence = np.zeros((4, self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        for i, image_path in enumerate(img_list_path):
            img = cv2.imread(frame_dir + image_path)
            frame_sequence[i] = cv2.resize(img, (self.img_shape[1], self.img_shape[0]))

        # audio, frame_sequence = torch.tensor(audio), torch.tensor(frame_sequence)
        frame_sequence = torch.tensor(frame_sequence, dtype=torch.float32)
        # return (int(label), txt, (fs, audio), frame_sequence)
        return (int(label), txt, frame_sequence)
    
if __name__ == "__main__" :
    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', resolution)
    print(dataset.__get_item__(0))