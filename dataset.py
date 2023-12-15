from torch.utils.data import Dataset
import pandas as pd
from scipy.io import wavfile
import os
import numpy as np
import cv2
from torchvision.transforms import ToTensor
from transformers import BertTokenizer

class Dataset(Dataset):
    def __init__(self, labels_file_path, frame_dir, audio_dir, txt_data_file_path, img_shape, txt_model='bert-base-multilingual-cased'):
        self.labels = pd.read__csv(labels_file_path)
        self.frame_dir = frame_dir
        self.audio_dir = audio_dir
        self.img_shape = img_shape
        self.txt_data = pd.read_csv(txt_data_file_path)
        self.transform = ToTensor()
        self.tokenizer = BertTokenizer.from_pretrained(txt_model)

    def __len__(self) :
        return len(self.labels)

    def __get_item__(self, index):
        label = self.labels["turn_after"][index]

        txt = self.txt_data[index]
        txt_tokens = self.tokenizer(txt)

        speaker = self.labels["speaker"][index]
        record = self.labels["dyad"][index][8:]
        audio_path = self.audio_dir + f"{record}/{record}_{speaker}_{index}.wav"
        audio = wavfile.read(audio_path)

        frame_dir = self.frame_dir + f"IPU_{index}/"
        img_list_path = os.listdir(frame_dir)

        frame_sequence = np.array((len(img_list_path), self.img_shape[0], self.img_shape[1], self.img_shape[2]))
        for i, image_path in enumerate(img_list_path):
            frame_sequence[i] = cv2.imread(frame_dir + image_path)

        audio, frame_sequence = self.transform(audio), self.transform(frame_sequence)

        return (label, txt_tokens, audio, frame_sequence)