import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from .tools import ChessBaord
from .data import Image, Result

class StereoAligner():
    def __init__(self, align_reference='Left', chessboard_size = (9, 6)):
        self.chessboard_size = chessboard_size
        self.chessboard = ChessBaord(chessboard_size)
        self.align_reference = align_reference

    def compute_alignment_mean(self, imgL, imgR):
        if self.align_reference == 'Left':
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

class StereoCalibrator():
    def __init__(self, device, chessboard_size = (9, 6)):
        images = {}
        for side in ['Left', 'Right']:
            images_dir = os.path.join('data', 'calibrate', device, side)
            images[side] = [
                os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                if f.lower().endswith('.jpg')
            ]
        self.images = images
        self.chessboard_size = chessboard_size
        self.chessboard = ChessBaord(chessboard_size)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    def _as_gray(self, img_path):
        image = cv2.imread(img_path)
        return image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def find_corners(self, imgL, imgR):
        """
        自動計算左右影像的棋盤格角點座標。
        回傳 (ret_left, corners_left), (ret_right, corners_right)
        """
        if len(imgL.shape) == 3:
            img_grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        else:
            img_grayL = imgL
        if len(imgR.shape) == 3:
            img_grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        else:
            img_grayR = imgR
        ret_left, corners_left = cv2.findChessboardCorners(img_grayL, self.chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(img_grayR, self.chessboard_size, None)
        if ret_left:
            corners_left = cv2.cornerSubPix(img_grayL, corners_left, (11,11), (-1,-1), self.chessboard.criteria)
        if ret_right:
            corners_right = cv2.cornerSubPix(img_grayR, corners_right, (11,11), (-1,-1), self.chessboard.criteria)
        return (ret_left, corners_left), (ret_right, corners_right)

    def fit_intrinsics(self, side):
        print(f"[StereoCalibrator] 開始標定 {side} 相機內參...")
        imgpoints = []
        for idx, img_path in enumerate(self.images[side]):
            img, img_gray = self._as_gray(img_path)
            ret, corners = cv2.findChessboardCorners(img_gray, self.chessboard_size, None)
            if ret:
                corners = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), self.criteria)
                imgpoints.append(corners)
        objpoints = self.chessboard.generate(samples = len(imgpoints))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f"[StereoCalibrator] {side} 相機內參標定完成，RMS投影誤差: {ret:.4f}")
        return ret, mtx, dist, rvecs, tvecs

    def fit(self):
        print("[StereoCalibrator] 開始執行雙目相機完整標定流程...")
        retL, mtxL, distL, rvecsL, tvecsL = self.fit_intrinsics('Left')
        retR, mtxR, distR, rvecsR, tvecsR = self.fit_intrinsics('Right')
        print("[StereoCalibrator] 內參標定完成，開始外參標定...")
        imgpointsL, imgpointsR = [], []
        for left_image, right_image in zip(self.images['Left'], self.images['Right']):
            imgL, img_grayL = self._as_gray(left_image)
            imgR, img_grayR = self._as_gray(right_image)
            ret_left, corners_left = cv2.findChessboardCorners(img_grayL, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(img_grayR, self.chessboard_size, None)
            if ret_left and ret_right:
                imgpointsL.append(cv2.cornerSubPix(img_grayL, corners_left, (11,11), (-1,-1), self.criteria))
                imgpointsR.append(cv2.cornerSubPix(img_grayR, corners_right, (11,11), (-1,-1), self.criteria))
        objpoints = self.chessboard.generate(samples = len(imgpointsL))

        flags = cv2.CALIB_FIX_INTRINSIC
        print("[StereoCalibrator] 執行 cv2.stereoCalibrate ...")
        retS, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtxL, distL, mtxR, distR, gray_left.shape[::-1],
            criteria=self.criteria, flags=flags
        )
        print(f"[StereoCalibrator] 雙目外參標定完成，RMS投影誤差: {retS:.4f}")
        print("[StereoCalibrator] 標定流程結束。")
        return {
            'retL': retL, 'mtxL': mtxL, 'distL': distL,
            'retR': retR, 'mtxR': mtxR, 'distR': distR,
            'retS': retS, 'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2, 'R': R, 'T': T, 'E': E, 'F': F
        }