from torch.utils.data import Dataset
import pandas as pd
from scipy.io import wavfile
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


resolution = (3,256,256)
number_of_frames = 4
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class Dataset(Dataset):
    def __init__(self, labels_file_path, frame_dir, audio_dir, txt_data_file_path, img_shape, preprocess = None):
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
            img = Image.open(frame_dir + image_path) 
            img = transforms.Pad(padding=256)(img)
            img = self.preprocess(img)
            frame_sequence[i] = img

        # audio, frame_sequence = torch.tensor(audio), torch.tensor(frame_sequence)
        frame_sequence = torch.tensor(frame_sequence, dtype=torch.float32)
        # return (int(label), txt, (fs, audio), frame_sequence)
        return (int(label), txt, frame_sequence)
    
if __name__ == "__main__" :
    dataset = Dataset('labels.csv', 'data/video/dataset_frame/', 'data/audio/samples/', 'txt_data.csv', resolution)
    print(dataset.__getitem__(0))