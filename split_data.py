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

        audio = data['dyad'][i]

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

def extract_video(df, input_video_path, output_video_path,video_path, nb_frame):

    # Create a VideoCapture object
    cap = cv2.VideoCapture(input_video_path)
    speaker1, speaker2 = video_path.split('_')
    speaker2 = speaker2[:-4]
    try :
        if math.isnan(speaker1) :
            speaker1 = "NA"
    except TypeError:
        pass
    try :
        if math.isnan(speaker2) :
            speaker2 = "NA"
    except TypeError:
        pass

    print(f'Extracting {speaker1} and {speaker2} ...')
    df = df[(df['dyad']== speaker1 + speaker2) | (df['dyad']==speaker2 + speaker1)]


    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        exit()

    # Get the frames per second (fps) and frame size
    fps = cap.get(cv2.CAP_PROP_FPS)

    for i in tqdm(range(df.shape[0])):
        line = df.iloc[i]
        start_time = line.start
        stop_time = line.stop
        start_frame = int(start_time * fps)
        stop_frame = int(stop_time * fps)
        if stop_frame - start_frame >= 4 :
            start_temp = stop_frame - nb_frame
        else :
            start_temp = start_frame
            
        name_ipu = f'IPU_{line.id}'  
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_temp)
        try:
            os.makedirs(output_video_path + name_ipu)
            num = 0
            
            for frame_num in range(start_temp, stop_frame):

                ret, frame = cap.read()

                if not ret:
                    print("Error: Unable to read frame.")
                    break

                # Write the frame to the output video
                output_name = output_video_path + name_ipu + "/" + f'{num}.jpeg'
                cv2.imwrite(output_name, frame)
                num+=1

        except FileExistsError:
            print("Folder of frames already exists !")

    # Release the VideoCapture and VideoWriter objects
    cap.release()

    print(f"Extracted video saved to {output_video_path}")

def split_and_save_video_ipu(data, input_videos_path,output_videos_path,nb_frame):
    if not os.path.exists(input_videos_path) :
        raise FileNotFoundError("Input videos path not found")

    if not os.path.exists(output_videos_path) :
        os.makedirs(output_videos_path)

    videos_path_cheese = os.listdir(input_videos_path + "cheese/")
    for video_path in videos_path_cheese :
        input_video_path = input_videos_path + "cheese/" + video_path
        extract_video(data, input_video_path, output_videos_path, video_path, nb_frame)

    videos_path_paco = os.listdir(input_videos_path + "paco/")
    for video_path in videos_path_paco :
        input_video_path = input_videos_path + "paco/" + video_path
        extract_video(data, input_video_path, output_videos_path, video_path, nb_frame)


def main(args):
    
    data = pd.read_csv(args.file_path)

    save_txt_data(data)
    print("Txt data saved")

    save_labels(data)
    print("Labels saved")

    split_and_save_audio_ipu(data, args.audio_dir, args.audio_sample_dir)
    print("Audio splitted and saved")

    split_and_save_video_ipu(data,args.video_dir, args.frame_dir,args.nb_frame)
    print("Video splitted and saved")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Process and analyze data.")
    parser.add_argument("-f", "--file_path", type=str, default="data.csv", help="Path to the data CSV file")
    parser.add_argument("--audio_dir", type=str, default="data/audio/1_channels_wav/", help="Directory for audio files")
    parser.add_argument("--audio_sample_dir", type=str, default="data/audio/samples/", help="Directory for audio samples")
    parser.add_argument("--video_dir", type=str, default="data/video/", help="Directory for video files")
    parser.add_argument("--frame_dir", type=str, default="data/video/dataset_frame/", help="Directory for video frames")
    parser.add_argument("--sampling_rate", type=int, default=4, help="Sampling rate for video frames")
    parser.add_argument("--nb_frame", type=int, default=4, help="Minimum frames for an IPU video")

    main(parser.parse_args())