import cv2
import subprocess
import dlib
import os
import argparse
args = parser = argparse.ArgumentParser()
parser.add_argument('--fullpath', type=str,default='/home/weiyb/GeneFacePlusPlus-main/composevideo/shangnan.mp4')
parser.add_argument('--headpath', type=str,default='/home/weiyb/GeneFacePlusPlus-main/composevideo/shangnanhead.mp4')
opt = parser.parse_args()
#参数列表
inputFile = opt.fullpath
headFile = opt.headpath
#最终生成的video的路径，默认输出在inputfile的上一级文件夹
outputFolder = os.path.join(inputFile,'../')
# 通常你会想要规范化路径，消除路径中的 '../'用于识别.
outputFolder = os.path.normpath(outputFolder)

#video_name
name = inputFile.split('/')[-1].split('.')[0]

# 初始化dlib的人脸检测器
detector = dlib.get_frontal_face_detector()
def get_face_coordinates(image):
    # cv读取的图片转为RGB格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('saved_image.jpg',rgb_image)
    # 使用dlib的人脸检测器检测人脸
    detections = detector(rgb_image)
    
    if len(detections) > 0:
        face = detections[0]
        # 计算并返回人脸中心点
        center_x = (face.left() + face.right()) // 2
        center_y = (face.top() + face.bottom()) // 2
        return center_x, center_y
    else:
        return None

# 读取视频
video_capture = cv2.VideoCapture(inputFile)
# 获取自定义帧的人脸坐标
diy = 1
diy_number = 1
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if diy_number==diy:
        print("第一帧的大小",frame.shape)
        face_coords = get_face_coordinates(frame)
        break
    diy_number += 1
     
if face_coords is not None:
    center_x, center_y = face_coords
    print("Center coordinates of the first detected face:", center_x, center_y)
else:
    print("No face detected in the first frame.")
targetWH=512
crop_size = targetWH//2  
#crop_size = 512 
start_x = max(center_x - crop_size, 0)
start_y = max(center_y - crop_size, 0)

from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip

# 读取训练完的视频和原视频
clip_A = VideoFileClip(inputFile)
clip_B = VideoFileClip(headFile)

# 将训练完视频的位置设定在原视频中的特定位置
clip_B = clip_B.set_position((start_x, start_y))

# 创建一个复合视频
final_clip = CompositeVideoClip([clip_A, clip_B], size=clip_A.size)
#由于Vscode上读取的视频没有声音！！！所以这里再次传入一下推理音频
# 读取B视频的音频
audio_B = AudioFileClip(headFile)

# 将A视频的音频替换为B视频的音频
final_clip = final_clip.set_audio(audio_B)

# 将最终视频的长度设置为音频的长度
final_clip = final_clip.subclip(0, audio_B.duration)

# 保存最终的视频
final_clip.write_videofile(f"{outputFolder}/{name}_compose.mp4")
