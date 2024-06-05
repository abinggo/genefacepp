
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip

def add_audio(video_path, audio_path, output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = VideoFileClip(audio_path).audio
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_path, codec='libx264')
def overlay_image(original_image, new_image, x, y):
    h, w = new_image.shape[:2]
    half_h, half_w = h // 2, w // 2

    # Calculate the region of interest (ROI) in the original image
    x1, y1 = max(0, x - half_w), max(0, y - half_h)
    x2, y2 = min(original_image.shape[1], x1 + w), min(original_image.shape[0], y1 + h)

    # Adjust new image's dimensions if ROI goes out of bounds
    new_x1, new_y1 = max(0, half_w - x), max(0, half_h - y)
    new_x2, new_y2 = new_x1 + (x2 - x1), new_y1 + (y2 - y1)

    # Overlay new image onto the original image
    original_image[y1:y2, x1:x2] = new_image[new_y1:new_y2, new_x1:new_x2]

    return original_image

def main():
    # Paths to your files
    original_video_path = '/home/weiyubin/SyncTalk/process/video/laotan.MP4'
    head_video_path = '/home/weiyubin/SyncTalk/process/xunlianhou.mp4'
    landmarks_file = '/home/weiyubin/SyncTalk/process/result/laotan.npy'
    output_video_path = '/home/weiyubin/SyncTalk/process/laotantiehui_noaud.mp4'
    final_video_path = '/home/weiyubin/SyncTalk/process/laotantiehui.mp4'

    # Load the landmarks from the npy file
    landmarks = np.load(landmarks_file)

    # Open the original video and the head video
    original_video = cv2.VideoCapture(original_video_path)
    head_video = cv2.VideoCapture(head_video_path)

    # Get the video properties
    fps = original_video.get(cv2.CAP_PROP_FPS)
    width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each (x, y) coordinate from the landmarks
    for i, (x, y) in enumerate(landmarks):
        # Read a frame from the original video and the head video
        ret1, original_frame = original_video.read()
        ret2, head_frame = head_video.read()

        # Check if we successfully read a frame from both videos
        if not ret1 or not ret2:
            break

        # Overlay the head frame onto the original frame
        result_frame = overlay_image(original_frame, head_frame, int(x), int(y))

        # Write the result frame to the output video
        out.write(result_frame)

        print(f"Processed frame {i}")

    # Release the VideoCapture and VideoWriter objects
    original_video.release()
    head_video.release()
    out.release()
    # Add audio from the head video to the output video
    add_audio(output_video_path, head_video_path, final_video_path)

    print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    main()
