import cv2
import numpy as np

class Image():
    def __init__(self, img, chessboard):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.chessboard = chessboard
        self.ret, self.corners = self._find_corners(self.img_gray)

    def _find_corners(self, img_gray):
        ret, corners = cv2.findChessboardCorners(img_gray, self.chessboard.size, None)
        if ret:
            corners = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), self.chessboard.criteria)
        else:
            print("[ImageReader] 未找到棋盤格。")
        return ret, corners

class Result():
    def __init__(self, imgL: Image, imgR: Image, alignment_mean=None):
        self.left = imgL
        self.right = imgR
        self.alignment_mean = alignment_mean
    
    def plot(self, threadhold=1, num_lines=16, color=None, thickness=2, scale=0.5):
        h, w = self.left.img.shape[:2]
        interval = h // (num_lines + 1)
        imgL_draw = self.left.img.copy()
        imgR_draw = self.right.img.copy()
        # 根據 alignment_mean 決定顏色
        if color is None:
            if self.alignment_mean is not None:
                if self.alignment_mean < threadhold:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
            else:
                color = (255, 0, 0)
        for i in range(1, num_lines + 1):
            y = i * interval
            cv2.line(imgL_draw, (0, y), (w, y), color, thickness)
            cv2.line(imgR_draw, (0, y), (w, y), color, thickness)
        imgL_small = cv2.resize(imgL_draw, (0, 0), fx=scale, fy=scale)
        imgR_small = cv2.resize(imgR_draw, (0, 0), fx=scale, fy=scale)
        merged = cv2.hconcat([imgL_small, imgR_small])
        # 在右下角繪製 mean 和 std
        if self.alignment_mean is not None:
            text = f"Pixel Difference: {self.alignment_mean:.2f}"
        else:
            text = "Chessboard not detected, cannot compute alignment."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness_text = 1
        color_text = (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness_text)
        x = merged.shape[1] - tw - 20
        y = merged.shape[0] - 20
        cv2.putText(merged, text, (x, y), font, font_scale, color_text, thickness_text, cv2.LINE_AA)
        return merged