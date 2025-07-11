import cv2
import numpy as np
from utils.stereo import StereoAligner, StereoCalibrator

class Capture():
    
def read(cap, shape):
    ret, frame = capL.read()
    if not ret:
        return False, np.random.randint(0, 256, shape, dtype=np.uint8)
    return ret, frame

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

aligner = StereoAligner(chessboard_size=(9, 6))
while True:
    read
    retL, frameL = capL.read()
    retR, frameR = capR.read()