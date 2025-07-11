import cv2
import numpy as np
from utils.stereo import StereoAligner, StereoCalibrator

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

aligner = StereoAligner(chessboard_size=(9, 6))
for 

"""
calibrator = StereoCalibrator('C270')
params = calibrator.fit()
K1, D1, K2, D2, R, T = params['K1'], params['D1'], params['K2'], params['D2'], params['R'], params['T']

imgL = cv2.imread('./data/image/left.jpg')
imgR = cv2.imread('./data/image/right.jpg')
aligner = Aligner(imgL, imgR)
aligner.fit()
cv2.imshow('Rectified Pair with Scanlines', aligner.plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
image_size = (imgL.shape[1], imgL.shape[0])

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
imgL_rect = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
imgR_rect = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# 建立 StereoSGBM 視差估算器（參數可微調）
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # 必須為16的倍數
    blockSize=5,
    P1=8*1*5**2,
    P2=32*1*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0
disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_visual = np.uint8(disparity_visual)

cv2.imshow('Disparity', disparity_visual)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""