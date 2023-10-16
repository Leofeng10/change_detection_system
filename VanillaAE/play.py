import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import pdb
import torch.utils.data as data


def main():


    all_video_frames = []

    vide_frames = glob.glob(os.path.join("C:/Users/Feng Zhunyi/Desktop/Drone-Anomaly-main/dataset/test/railway", '*.jpg'))

    if len(all_video_frames) == 0:
        all_video_frames = vide_frames
    else:
        all_video_frames += vide_frames

    i = 0
    for frame in all_video_frames:
        image_decoded = cv2.imread(frame)
        cv2.imwrite("C:/Users/Feng Zhunyi/Desktop/focal-frequency-loss-master/datasets/celeba/img_align_celeba/" + str(i).zfill(6) + ".jpg", image_decoded)
        i += 1





if __name__ == '__main__':
    main()