# video to frames and resize to 256
# input: video files, *.mp4
# output: frame files, *.jpg

# By Pu Jin 2020.11.30

#! encoding: UTF-8

import os
import cv2

import glob

videos_src_path = 'D:\\Files\\MUC\\Anomaly detection\\dataset\\NEW\\test\\09\\'
videos_save_path = 'D:\\Files\\MUC\\Anomaly detection\\dataset\\dataset3\\test\\'


# videos = glob.glob(os.path.join("C:/Users/Feng Zhunyi/Desktop/Drone-Anomaly-main/dataset/testing/", '*'))
# print(len(videos))
# if os.path.isdir(videos[0]):
#     all_video_frames = []
#     videos.sort(key=lambda x: int(x[-3:]))
#     for video in videos:
#         print(video)
#         vide_frames = glob.glob(os.path.join(video, '*.jpg'))
#         # print(vide_frames)
#         # for i in vide_frames:
#         #     print(i[len(vide_frames[0])-8:len(vide_frames[0])-4])
#         # print(vide_frames[0][len(vide_frames[0])-8:len(vide_frames[0])-4])
#         vide_frames.sort(key=lambda x: int(x[len(x) - 8:len(x) - 4]))
#         frame_count = 0
#         for frame in vide_frames:
#             img = cv2.imread(frame)
#             frame_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#             cv2.imwrite("C:/Users/Feng Zhunyi/Desktop/Drone-Anomaly-main/dataset/test/" + video[len(video)-3:len(video)] + "/" + "%d.jpg" % frame_count, frame_resized)
#             frame_count += 1

# videos = os.listdir(videos_src_path)
# videos = filter(lambda x: x.endswith('mp4'), videos)

# for each_video in videos:
#
#     # get the name of each video, and make the directory to save frames
#     each_video_name, _ = each_video.split('.')
#     os.mkdir(videos_save_path + each_video_name)
#
#     each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '\\'
#
#     # get the full path of each video, which will open the video tp extract frames
#     each_video_full_path = os.path.join(videos_src_path, each_video)
#
#     cap = cv2.VideoCapture(each_video_full_path)
#     frame_count = 0
#     success = True
#     while success:
#         success, frame = cap.read()
#         # params = []
#         # params.append(cv.CV_IMWRITE_PXM_BINARY)
#         # params.append(1)
#         if success:
#             frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
#             cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, frame_resized)
#         frame_count = frame_count + 1
#
#     cap.release()

print(cv2.getBuildInformation())

each_video_name = 'railway'
# os.mkdir("C:/Users/Feng Zhunyi/Desktop/Drone-Anomaly-main/dataset/train/" + each_video_name)

each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

# get the full path of each video, which will open the video tp extract frames
each_video_full_path = "C:/Users/Feng Zhunyi/Desktop/unseen_bottle.mp4"
print(each_video_full_path)
cap = cv2.VideoCapture(each_video_full_path)
print(cap.isOpened())
frame_count = 0
success = True
while success:
    print(frame_count)
    success, frame = cap.read()
    print(success)
    # params = []
    # params.append(cv.CV_IMWRITE_PXM_BINARY)
    # params.append(1)
    if success:
        print("success")
        frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite("C:/Users/Feng Zhunyi/Desktop/focal-frequency-loss-master/datasets/celeba/unseen_bottle/" + "%d.jpg" % frame_count, frame_resized)
    frame_count = frame_count + 1

cap.release()