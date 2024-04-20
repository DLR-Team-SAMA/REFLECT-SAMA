import cv2
import numpy as np


kf_file = 'dataset_test/keyframe.txt'
f = open(kf_file, 'r')
lines = f.read()
lines = lines.split('\n')
# to int
lines = [int(i) for i in lines]
f.close()
print(lines)

# read video
cap = cv2.VideoCapture('dataset_test/make_coffee.mp4')

for i in lines:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(0)