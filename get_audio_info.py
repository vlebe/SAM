import os
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd

audio_folder = "data/audio/samples/"

records = os.listdir(audio_folder)

audio_times = []
to_remove = []

for record in tqdm(records) :
    record_path = audio_folder + record

    for audio_file in os.listdir(record_path) :
        fs, audio = wavfile.read(record_path + "/" + audio_file)
        audio_length = len(audio) / fs
        if audio_length <= 0.15 :
            print(audio_length)
            print(int(audio_file.split("_")[2].split(".")[0]))
            to_remove.append(int(audio_file.split("_")[2].split(".")[0]))
        audio_times.append(audio_length)

# data = pd.read_csv("data.csv")
# data = data[~data["id"].isin(to_remove)].reset_index()
# data["id"] = data.index

# data.to_csv("data.csv", index=False)

print("Total audio files: ", len(audio_times))
print("Average audio length: ", sum(audio_times) / len(audio_times))
print("Max audio length: ", max(audio_times))
print("Min audio length: ", min(audio_times))