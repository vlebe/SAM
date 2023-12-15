import cv2
import sys
import pandas as pd
import os

excel_file_path = 'data/labels.xlsx'
nb_frame = 4


def extract_video(input_video_path, output_video_path,video_path):
    global excel_file_path, number_frame
    # Create a VideoCapture object
    cap = cv2.VideoCapture(input_video_path)
    df = pd.read_excel(excel_file_path)
    speaker1, speaker2 = video_path.split('_')
    speaker2 = speaker2[:-4]
    print(f'speakers : {speaker1} and {speaker2}')
    df = df[(df['dyad']==speaker1 + speaker2) | (df['dyad']==speaker2 + speaker1)]
    input()


    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open input video file.")
        exit()

    # Get the frames per second (fps) and frame size
    fps = cap.get(cv2.CAP_PROP_FPS)



    for i in range(df.shape[0]):
        line = df.iloc[i]
        print(line[line.index[0]])
        start_time = line.start
        stop_time = line.stop
        start_frame = int(start_time * fps)
        stop_frame = int(stop_time * fps)
        if stop_frame - start_frame >= 4 :
            start_temp = stop_frame - nb_frame
        name_ipu = f'IPU_{str(line[line.index[0]]) + ".mp4"}'
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_temp)
        os.makedirs(output_video_path + name_ipu)
        num = 0

        # frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
        # out = cv2.VideoWriter(output_video_path + name_ipu + "/" + f'{name_ipu}.mp4', fourcc, fps, frame_size)
        
        for frame_num in range(start_temp, stop_frame):

            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read frame.")
                break

            # Write the frame to the output video
            output_name = output_video_path + name_ipu + "/" + f'{num}.jpeg'
            cv2.imwrite(output_name, frame)
            # out.write(frame)
            num+=1

        print(f'{name_ipu} extracted')
        # out.release()
    input()

    # Release the VideoCapture and VideoWriter objects
    cap.release()

    print(f"Extracted video saved to {output_video_path}")

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py input_videos_path output_videos_path")
        exit()

    # Extract command-line arguments
    input_videos_path = sys.argv[1]
    output_videos_path = sys.argv[2]

    try:
        # Check if the folder exists
        if os.path.exists(input_videos_path) and os.path.isdir(input_videos_path):
            print(f"The folder '{input_videos_path}' exists.")
        else:
            print(f"The folder '{input_videos_path}' does not exist or is not a directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        # Check if the folder exists
        if os.path.exists(output_videos_path) and os.path.isdir(output_videos_path):
            print(f"The folder '{output_videos_path}' exists.")
        else:
            print(f"The folder '{output_videos_path}' does not exist or is not a directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

    videos_path = os.listdir(input_videos_path)
    for video_path in videos_path :
        input_video_path = input_videos_path + video_path
        print(video_path)
        extract_video(input_video_path, output_videos_path, video_path)
        input()
