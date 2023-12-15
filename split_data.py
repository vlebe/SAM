import pandas as pd
import os
from scipy.io import wavfile
from tqdm import tqdm
import cv2
import os
import math
import argparse 

def save_txt_data(data) :
    txt_data = data[["text"]]
    txt_data.to_csv("txt_data.csv", index=False, encoding="utf-8-sig")

def save_labels(data) :
    labels = data[["speaker", "dyad","turn_after"]]
    labels.to_csv("labels.csv", index=False, encoding="utf-8-sig")

def split_and_save_audio_ipu(data, audio_dir, audio_sample_dir) :
    if not os.path.exists(audio_sample_dir) :
        os.makedirs(audio_sample_dir)
        
    df = data[['speaker', "start", 'stop', 'dyad']]
    
    for i, tf, ts in tqdm(zip(df.index, df.start, df.stop)):

        if ts - tf > 2 :
            tf = ts - 2

        audio = data['dyad'][i][8:]

        os.makedirs(audio_sample_dir + f"{audio}/", exist_ok=True)

        speaker = data['speaker'][i]
        if type(speaker) != str :
            speaker = "NA"

        file_path = audio_dir + f"{audio}_{speaker}.wav"

        fs, x = wavfile.read(file_path)
        start = int(tf * fs)
        stop = int(ts * fs)
        audio_sample = x[start:stop]

        wavfile.write(audio_sample_dir + f"{audio}/" + f"{audio}_{speaker}_{i}.wav", fs, audio_sample)

def split_video_into_frames(video_path, output_folder, format='jpeg', sr=1, tstart=0, tstop=None):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_rate = int(cap.get(5))  # Frame rate
    total_frames = int(cap.get(7))  # Total number of frames

    # Calculate the frame interval based on the desired sampling rate
    frame_interval = int(frame_rate / sr)

    # Convert time values to frame indices
    frame_start = int(tstart * frame_rate)
    frame_stop = int(tstop * frame_rate) if tstop is not None else total_frames

    # Ensure the frame indices are within valid range
    frame_start = max(0, min(frame_start, total_frames - 1))
    frame_stop = max(frame_start, min(frame_stop, total_frames))

    # Start processing frames
    for i in range(frame_start, frame_stop, frame_interval):
        # Set the frame position
        cap.set(1, i)

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        # Define the output file path
        output_file_path = os.path.join(output_folder, f'frame_{i:04d}.{format}')

        # Save the frame
        cv2.imwrite(output_file_path, frame)

    # Release the video capture object
    cap.release()

def split_and_save_video_frames(data, video_dir, frame_dir, sr, min_duration) :
    if not os.path.exists(frame_dir) :
        os.makedirs(frame_dir)

    df = data[['speaker', "start", 'stop', 'dyad']]
    for i, tf, ts in tqdm(zip(df.index, df.start, df.stop)):
        video = data['dyad'][i][8:]
        os.makedirs(frame_dir + f"{video}/", exist_ok=True)

        speaker = data['speaker'][i]
        if math.isnan(speaker) :
            speaker = "NA"

        video_path = video_dir + f"{video}.mp4"
        output_folder = frame_dir + f"{video}/{speaker}/"
        format = "jpeg"  # or "png"

        if ts-tf > min_duration:
            tf = tf - min_duration

        split_video_into_frames(video_path, output_folder, format=format, sr=sr, tstart=tf, tstop=ts)

def main(args):
    
    data = pd.read_csv(args.file_path)

    save_txt_data(data)
    print("Txt data saved")

    save_labels(data)
    print("Labels saved")

    split_and_save_audio_ipu(data, args.audio_dir, args.audio_sample_dir)
    print("Audio splitted and saved")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Process and analyze data.")
    parser.add_argument("-f", "--file_path", type=str, default="data.csv", help="Path to the data CSV file")
    parser.add_argument("--audio_dir", type=str, default="data/audio/1_channels_wav/", help="Directory for audio files")
    parser.add_argument("--audio_sample_dir", type=str, default="data/audio/samples/", help="Directory for audio samples")
    parser.add_argument("--video_dir", type=str, default="data/video/dataset_video/", help="Directory for video files")
    parser.add_argument("--frame_dir", type=str, default="data/video/dataset_frame/", help="Directory for video frames")
    parser.add_argument("--sampling_rate", type=int, default=4, help="Sampling rate for video frames")
    parser.add_argument("--min_duration", type=int, default=2, help="Minimum duration for video frames")
    
    main(parser.parse_args())