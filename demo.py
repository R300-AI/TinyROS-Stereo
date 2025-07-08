import cv2
from stereo import StereoConverter

print('開啟左相機 (Camera 0)...')
cap = cv2.VideoCapture(0)
ret, left_img = cap.read()
cap.release()
if not ret:
    raise IOError('無法從 Camera 0 取得影像')
print('左影像擷取完成')

print('開啟右相機 (Camera 1)...')
cap = cv2.VideoCapture(1)
ret, right_img = cap.read()
cap.release()
if not ret:
    raise IOError('無法從 Camera 1 取得影像')
print('右影像擷取完成')

# 轉成灰階，避免 StereoBM 報錯
left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

#left_img = cv2.imread("data/holopix50k_images/test/00000_left.jpg", cv2.IMREAD_GRAYSCALE)
#right_img = cv2.imread("data/holopix50k_images/test/00000_right.jpg", cv2.IMREAD_GRAYSCALE)
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