import cv2
import numpy as np

# --- 1. Camera Setup and Capture ---
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# --- 2. Camera Calibration (requires a set of calibration images) ---
objpoints = [] # 3D points in real world space
imgpoints_left = [] # 2D points in image plane for left camera
imgpoints_right = [] # 2D points in image plane for right camera

ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(...)
et_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(...)

ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(...)

# --- 3. Stereo Rectification ---
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, img_size, R, T)

map1_left, map2_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, img_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, img_size, cv2.CV_16SC2)

# --- 4. Disparity Map Generation (in a loop for live stream) ---
while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    undistorted_rectified_left = cv2.remap(frame_l, map1_left, map2_left, cv2.INTER_LINEAR)
    undistorted_rectified_right = cv2.remap(frame_r, map1_right, map2_right, cv2.INTER_LINEAR)

    stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15) # Adjust parameters
    disparity = stereo.compute(cv2.cvtColor(undistorted_rectified_left, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(undistorted_rectified_right, cv2.COLOR_BGR2GRAY))

    # Normalize disparity for visualization
    disparity_normalized = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('Disparity Map', disparity_normalized)

# --- 5. Depth Map Calculation (optional, from disparity) ---
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    # Access depth from points_3D[:, :, 2]