import cv2
import numpy as np
from utils.stereo import Calibrator
import itertools
import os

calibrator = Calibrator()
params = calibrator.fit()
K1, D1, K2, D2, R, T = params['K1'], params['D1'], params['K2'], params['D2'], params['R'], params['T']

# 讀取一組成對校正影像
imgL = cv2.imread(calibrator.images['left'][0])
imgR = cv2.imread(calibrator.images['right'][0])
image_size = (imgL.shape[1], imgL.shape[0])

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
imgL_rect = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
imgR_rect = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

def run_sgbm_grid_search(imgL_rect, imgR_rect):
    # 淨空 logs 資料夾
    import shutil
    from tqdm import tqdm
    if os.path.exists('logs'):
        shutil.rmtree('logs')
    os.makedirs('logs', exist_ok=True)
    # 參數空間（每個tick都遍歷，不跳躍）
    numDisparities_list = list(range(16, 257, 16)) # 16~256, 間隔16
    blockSize_list = list(range(3, 12, 2))        # 3~11, 奇數
    uniquenessRatio_list = list(range(0, 21, 4))  # 0~20, 間隔1
    speckleWindowSize_list = list(range(0, 201, 25)) # 0~200, 間隔10
    speckleRange_list = list(range(0, 33, 4))     # 0~32, 間隔1
    preFilterCap_list = list(range(1, 64, 4))     # 1~63, 間隔1
    mode_dict = {
        'SGBM': cv2.STEREO_SGBM_MODE_SGBM,
        'HH': cv2.STEREO_SGBM_MODE_HH,
        'SGBM_3WAY': cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        'HH4': cv2.STEREO_SGBM_MODE_HH4
    }
    # 其他固定參數
    minDisparity = 0
    disp12MaxDiff = 0
    for mode_name, mode in mode_dict.items():
        subdir = os.path.join('logs', mode_name)
        os.makedirs(subdir, exist_ok=True)
        param_combinations = list(itertools.product(
            numDisparities_list, blockSize_list, uniquenessRatio_list, speckleWindowSize_list, speckleRange_list, preFilterCap_list))
        with tqdm(param_combinations, desc=f"GridSearch {mode_name}") as pbar:
            for params in pbar:
                numDisparities, blockSize, uniquenessRatio, speckleWindowSize, speckleRange, preFilterCap = params
                P1 = 8 * 3 * blockSize ** 2
                P2 = 32 * 3 * blockSize ** 2
                stereo = cv2.StereoSGBM_create(
                    minDisparity=minDisparity,
                    numDisparities=numDisparities,
                    blockSize=blockSize,
                    P1=P1,
                    P2=P2,
                    disp12MaxDiff=disp12MaxDiff,
                    uniquenessRatio=uniquenessRatio,
                    speckleWindowSize=speckleWindowSize,
                    speckleRange=speckleRange,
                    preFilterCap=preFilterCap,
                    mode=mode
                )
                disparity = stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0
                disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                disparity_visual = np.uint8(disparity_visual)
                # 儲存結果到子資料夾
                fname = f"{subdir}/disparity_d{numDisparities}_b{blockSize}_u{uniquenessRatio}_sw{speckleWindowSize}_sr{speckleRange}_pf{preFilterCap}.png"
                cv2.imwrite(fname, disparity_visual)
                pbar.set_postfix({"file": os.path.basename(fname)})
                #print(f"Saved: {fname}")

run_sgbm_grid_search(imgL_rect, imgR_rect)