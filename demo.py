from itertools import combinations
import numpy as np
import math, os, cv2
from utils import draw_chessboard, imshow, plot_corresponding_points, plot_pts3d_gif
    
CHESSBOARD_SIZE = (9, 6)
OBJP = draw_chessboard(CHESSBOARD_SIZE)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 64, 1e-6)

left_dir = os.path.join('data', 'calibrate', 'C270', 'Left')
right_dir = os.path.join('data', 'calibrate', 'C270', 'Right')
left_images = sorted([os.path.join(left_dir, f) for f in os.listdir(left_dir) if f.lower().endswith('.jpg')])
right_images = sorted([os.path.join(right_dir, f) for f in os.listdir(right_dir) if f.lower().endswith('.jpg')])

# 單目標定
imgpoints = []
valid_imgs = []
corners_list = []

for idx, fname in enumerate(left_images + right_images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), CRITERIA)
        imgpoints.append(corners)
        valid_imgs.append(img)
        corners_list.append(corners)

objpoints = [OBJP for _ in imgpoints]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f'RMS Projection Error: {ret}')

# 雙目標定
imgpoints_left, imgpoints_right = [], []
for left_image, right_image in zip(left_images, right_images):
    img_left = cv2.imread(left_image)
    img_right = cv2.imread(right_image)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, None)
    if ret_left and ret_right:
        imgpoints_left.append(cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), CRITERIA))
        imgpoints_right.append(cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), CRITERIA))
objpoints = [OBJP for _ in imgpoints_left]

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx, dist, mtx, dist, gray.shape[::-1],
    criteria=CRITERIA, 
    flags=cv2.CALIB_USE_INTRINSIC_GUESS
)
print(f'RMS Projection Error: {ret}')

def scale_points_to_square(points, img_shape):
    """
    將影像座標 points 依照 img_shape (height, width) 縮放成 1:1 正方形比例
    points: list of (x, y)
    img_shape: (height, width)
    回傳: list of (x', y')，已正規化到最大邊長為1的正方形
    """
    h, w = img_shape
    scale = max(h, w)
    return [((x / w) * scale, (y / h) * scale) for (x, y) in points]
# 測試轉換
pts_left = [(690, 179), (1018, 179), (695, 647), (1275, 652)]
pts_right = [(470, 189), (787, 204), (166, 630), (708, 675)]

pts_left_np = np.array(pts_left, dtype=np.float32).T
pts_right_np = np.array(pts_right, dtype=np.float32).T
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, gray.shape[::-1], R, T)
pts4d = cv2.triangulatePoints(P1, P2, pts_left_np, pts_right_np)
pts3d = (pts4d[:3] / pts4d[3]).T
print('四個物體的3D位置:')
for i, pt in enumerate(pts3d):
    print(f'Point {i+1}:', pt)

print('\n四個物體兩兩之間的距離:')
for (i, pt1), (j, pt2) in combinations(enumerate(pts3d), 2):
    dist = np.linalg.norm(pt1 - pt2)
    print(f'Point {i+1} <-> Point {j+1}: {dist:.4f} cm')


pts_left = scale_points_to_square(pts_left, gray.shape)
pts_right = scale_points_to_square(pts_right, gray.shape)
pts_left_np = np.array(pts_left, dtype=np.float32).T
pts_right_np = np.array(pts_right, dtype=np.float32).T
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, gray.shape[::-1], R, T)
pts4d = cv2.triangulatePoints(P1, P2, pts_left_np, pts_right_np)
pts3d = (pts4d[:3] / pts4d[3]).T
print('四個物體的3D位置:')
for i, pt in enumerate(pts3d):
    print(f'Point {i+1}:', pt)

print('\n四個物體兩兩之間的距離:')
for (i, pt1), (j, pt2) in combinations(enumerate(pts3d), 2):
    dist = np.linalg.norm(pt1 - pt2)
    print(f'Point {i+1} <-> Point {j+1}: {dist:.4f} cm')