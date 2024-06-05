import os
import cv2
import face_alignment
import numpy as np
import torch
from tqdm import tqdm
import argparse
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import os
def sort_key(filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Split the name into prefix and number
    prefix, number = name.rsplit('_', 1)
    
    # Convert the number to an integer and return it
    return int(number)
def save_landmarks(landmarks, output_landmarks_path):
    np.save(output_landmarks_path, landmarks)
    print(f"Landmarks saved to {output_landmarks_path}")

def process_video_and_crop_frames(video_path, output_folder, frames_per_second=25, crop_size=512):
    
    name = video_path.split('/')[-1].split('.')[0]
    landmarks_file=output_image_path = os.path.join(output_folder, f"{name}.npy")
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the face alignment pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)

    # Load the video
    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval to pick frames
    interval = max(1, fps // frames_per_second)
    
    frame_idx = 0
    landmarks_list = []

    with tqdm(total=total_frames, desc="Processing video frames", unit="frame") as pbar:
        while frame_idx < total_frames:
            # Set the video position to the current frame index
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            
            if not success:
                print(f"Error reading frame {frame_idx}")
                frame_idx += interval
                continue
            
            # Detect landmarks
            landmarks = fa.get_landmarks(frame)
            
            if landmarks is None or len(landmarks) == 0:
                print(f"No faces detected in frame {frame_idx}")
                frame_idx += interval
                continue
            
            # Get the 30th landmark (index 29)
            landmark_30 = landmarks[0][29]
            landmarks_list.append(landmark_30)
            
            # Calculate the crop region
            x, y = int(landmark_30[0]), int(landmark_30[1])
            half_size = crop_size // 2
            x1, y1 = max(0, x - half_size), max(0, y - half_size)
            x2, y2 = x1 + crop_size, y1 + crop_size
            
            # Ensure the crop region is within the frame boundaries
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
                x1 = x2 - crop_size
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]
                y1 = y2 - crop_size
            
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Save the cropped frame
            # Ensure output frame folder exists
            frame_folder = os.path.join(output_folder, 'frame')
            os.makedirs(frame_folder, exist_ok=True)

            # Save the cropped frame
            output_image_path = os.path.join(frame_folder, f'cropped_frame_{frame_idx}.jpg')
            cv2.imwrite(output_image_path, cropped_frame)

            
            frame_idx += interval
            pbar.update(interval)
    # Save landmarks to a file
    save_landmarks(np.array(landmarks_list), landmarks_file)
    video_capture.release()
    
    # Extract audio from the original video
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(os.path.join(output_folder, 'audio.wav'))
    
    
   
    frame_folder = os.path.join(output_folder, 'frame')
    # Get a list of all image files in the folder
    image_files = os.listdir(frame_folder)

    # Sort the list in the order you want
    image_files.sort(key=sort_key)

    # Create full paths for each image file
    image_paths = [os.path.join(frame_folder, filename) for filename in image_files]

    # Create a video from the image sequence
    video_without_audio = ImageSequenceClip(image_paths, fps=25)


    video_without_audio.write_videofile(os.path.join(output_folder, f"{name}_noaud.mp4"))

    # Add audio to the video
    video_with_audio = video_without_audio.set_audio(audio)
    #video_with_audio.write_videofile(os.path.join(output_folder, f"{name}.mp4"))
    video_with_audio.write_videofile(os.path.join(output_folder, f"{name}.mp4"), codec='libx264')


def run(input_path,outvideo_path):
    # Process video and crop frames
    process_video_and_crop_frames(input_path, outvideo_path)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_path',type=str,default='/home/weiyubin/SyncTalk/process/video/laotan.MP4')#输入的视频路径
    parser.add_argument('--out_folder',type=str,default='/home/weiyubin/SyncTalk/process/result')#输出的视频路径
    args=parser.parse_args()
    run(args.input_path,args.out_folder)
    #data = np.load('/home/yanxiaole/Aigc/SyncTalk/scripts/crop/save_landmarks.npy')
    #print(data.shape)


