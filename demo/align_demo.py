import cv2
import numpy as np
import time
from utils.tools import Camera
from utils.stereo import Aligner, Calibrator

capL, capR = Camera(5), Camera(7)
aligner = StereoAligner(chessboard_size=(9, 6))
while True:
    frameL = capL.read()
    frameR = capR.read()
    result = aligner.fit(frameL, frameR)
    merged = result.plot(threadhold=2.0)
    cv2.imshow('Stereo Alignment Streaming', merged)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.estimateAffinePartial2D
capL.release()
capR.release()
cv2.destroyAllWindows()

