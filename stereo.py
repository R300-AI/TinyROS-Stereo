import os
import cv2
import numpy as np

class StereoConverter:
    def __init__(self, 
                 focal_length, # 焦距(像素)：可用相機標定工具(如OpenCV calibration)取得，或用已知物體距離與像素長度反推
                 baseline,     # 基線距離(公尺)：兩個相機鏡頭中心點的實際物理距離，可用尺量測
                 numDisparities, # 最大視差範圍(像素)：建議設為16的倍數，略大於場景最大預期視差，解析度高/物體近時需大一點
                 blockSize     # 區塊大小(像素)：必須為奇數，5~21間調整，雜訊多用大值，細節多用小值
    ):
        self.focal_length = focal_length
        self.baseline = baseline
        self.stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    def compute_disparity(self, left_img, right_img):
        disparity = self.stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        return disparity

    def compute_depth(self, disparity):
        disparity[disparity == 0] = 0.1
        depth = (self.focal_length * self.baseline) / disparity
        return depth

    def transform(self, left_img, right_img, verbose=True):
        disparity = self.compute_disparity(left_img, right_img)
        depth = self.compute_depth(disparity)
        if verbose:
            disp_img = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8)
            depth_img = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            if not os.path.exists('logs'):
                os.makedirs('logs')
            left_color = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            right_color = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
            disp_color = cv2.applyColorMap(disp_img, cv2.COLORMAP_JET)
            depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            top = np.hstack([left_color, right_color])
            bottom = np.hstack([disp_color, depth_color])
            merged = np.vstack([top, bottom])
            cv2.imwrite(f'logs/stereo_transform_results.png', merged)
        return disparity, depth

if __name__ == "__main__":
    import cv2

    left_img = cv2.imread("data/holopix50k_images/test/00000_left.jpg", cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread("data/holopix50k_images/test/00000_right.jpg", cv2.IMREAD_GRAYSCALE)
    stereo = StereoConverter(
        focal_length=700, 
        baseline=0.06, 
        numDisparities=64, 
        blockSize=15
    )
    disparity, depth = stereo.transform(left_img, right_img, verbose=True)
    print('Disparity/Depth computed and images saved to logs/')
