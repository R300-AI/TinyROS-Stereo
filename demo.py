import cv2
from stereo import StereoConverter

left_img = cv2.imread("data/demo_left.jpg", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("data/demo_right.jpg", cv2.IMREAD_GRAYSCALE)
print('建立 StereoConverter 物件...')
stereo = StereoConverter(
    focal_length=700, 
    baseline=0.38, 
    numDisparities=64, 
    blockSize=15
)
print('開始計算 disparity/depth...')
disparity, depth = stereo.transform(left_img, right_img, verbose=True)
print('Disparity/Depth computed and images saved to logs/')