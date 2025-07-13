import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from .tools import ChessBaord
from .data import Image, Result

class StereoBase():
    def __init__(self, align_reference='left', chessboard_size = (9, 6)):
        self.chessboard_size = chessboard_size
        self.chessboard = ChessBaord(chessboard_size)
        self.align_reference = align_reference

class Aligner(StereoBase):
    def __init__(self, align_reference='left', chessboard_size = (9, 6)):
        super().__init__(align_reference=align_reference, chessboard_size=chessboard_size)

    def compute_alignment_mean(self, imgL, imgR):
        if self.align_reference == 'left':
            diffs = imgL.corners[:, 0, 1] - imgR.corners[:, 0, 1]
        else:
            diffs = imgR.corners[:, 0, 1] - imgL.corners[:, 0, 1]
        return np.mean(np.abs(diffs)), diffs

    def fit(self, imgL, imgR):
        imgL = Image(imgL, self.chessboard)
        imgR = Image(imgR, self.chessboard)
        if not imgL.ret or not imgR.ret:
            return Result(imgL, imgR, None)
        mean, diffs = self.compute_alignment_mean(imgL, imgR)
        return Result(imgL, imgR, mean)

class Calibrator(StereoBase):
    def __init__(self, align_reference='left', chessboard_size = (9, 6)):
        super().__init__(align_reference=align_reference, chessboard_size=chessboard_size)
        images = {}
        for side in ['Left', 'Right']:
            images_dir = os.path.join('data', 'calibrate', side)
            images[side] = [
                os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                if f.lower().endswith('.jpg')
            ]
        # 只保留left和right資料夾中同名的成對樣本
        left_dir = os.path.join('data', 'calibrate', 'left')
        right_dir = os.path.join('data', 'calibrate', 'right')
        left_files = set(f for f in os.listdir(left_dir) if f.lower().endswith('.jpg'))
        right_files = set(f for f in os.listdir(right_dir) if f.lower().endswith('.jpg'))
        paired_files = sorted(left_files & right_files)
        self.images = {
            'left': [os.path.join(left_dir, f) for f in paired_files],
            'right': [os.path.join(right_dir, f) for f in paired_files]
        }
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    def fit_intrinsics(self, side):
        print(f"[StereoCalibrator] 開始標定 {side} 相機內參...")
        imgpoints = [Image(img_path, self.chessboard).corners for img_path in self.images[side]]
        objpoints = self.chessboard.sampling(size = len(self.images[side]))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, 
            Image(self.images[self.align_reference][0], self.chessboard).img_gray.shape[::-1], 
            None, None
        )
        print(f"[StereoCalibrator] {side} 相機內參標定完成，RMS投影誤差: {ret:.4f}")
        return ret, mtx, dist, rvecs, tvecs

    def fit(self):
        print("[StereoCalibrator] 開始執行雙目相機完整標定流程...")
        retL, mtxL, distL, rvecsL, tvecsL = self.fit_intrinsics('left')
        retR, mtxR, distR, rvecsR, tvecsR = self.fit_intrinsics('right')
        print("[StereoCalibrator] 內參標定完成，開始外參標定...")
        imgpointsL = [Image(img_path, self.chessboard).corners for img_path in self.images['left']]
        imgpointsR = [Image(img_path, self.chessboard).corners for img_path in self.images['right']]
        objpoints = self.chessboard.sampling(size = len(self.images[self.align_reference]))

        print("[StereoCalibrator] 執行 Stereo Calibrate ...")
        retS, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, 
            Image(self.images[self.align_reference][0], self.chessboard).img_gray.shape[::-1],
            criteria=self.criteria, flags=cv2.CALIB_FIX_INTRINSIC
        )
        print(f"[StereoCalibrator] 雙目外參標定完成，RMS投影誤差: {retS:.4f}")
        print("[StereoCalibrator] 標定流程結束。")
        return {
            'retL': retL, 'mtxL': mtxL, 'distL': distL,
            'retR': retR, 'mtxR': mtxR, 'distR': distR,
            'retS': retS, 'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2, 'R': R, 'T': T, 'E': E, 'F': F
        }