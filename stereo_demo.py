import cv2
import numpy as np
import yaml
import os

# 讀取相機參數
# 從 yaml 檔案載入相機標定參數（如焦距、基線、主點等）
def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

# 計算視差圖
# 使用 OpenCV 的 StereoBM 方法計算左右影像的視差圖
def compute_disparity(left_img, right_img):
    # 可根據需求調整參數
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    return disparity

# 根據視差和相機參數計算深度
# disparity: 視差圖, focal_length: 焦距(像素), baseline: 基線距離(公尺)
def disparity_to_depth(disparity, focal_length, baseline):
    # 避免除以0
    disparity[disparity == 0] = 0.1
    depth = (focal_length * baseline) / disparity
    return depth

if __name__ == "__main__":
    # 設定檔案路徑
    left_img_path = "data/holopix50k_images/test/00000_left.jpg"
    right_img_path = "data/holopix50k_images/test/00000_right.jpg"
    cam_param_path = "config/camera_params.yaml"

    # 讀取影像（灰階）
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None:
        print("影像讀取失敗，請確認路徑正確。")
        exit(1)

    # 讀取相機參數
    params = load_camera_params(cam_param_path)
    focal_length = params['focal_length']  # 焦距(像素)
    baseline = params['baseline']          # 基線距離(公尺)
    fx = params.get('fx', focal_length)    # x方向焦距
    fy = params.get('fy', focal_length)    # y方向焦距
    cx = params.get('cx', left_img.shape[1] // 2)  # 主點x
    cy = params.get('cy', left_img.shape[0] // 2)  # 主點y

    # 計算視差圖
    disparity = compute_disparity(left_img, right_img)
    # 將視差圖正規化為 0~255 並存檔
    disp_img = ((disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255).astype(np.uint8)
    cv2.imwrite("disparity_demo.png", disp_img)

    # 計算深度圖（單位：公尺）
    depth = disparity_to_depth(disparity, focal_length, baseline)
    np.save("depth_demo.npy", depth)

    # 查詢某個像素的 3D 座標 (單位：公尺)
    u, v = 100, 100  # 你可以改成你想查的像素座標
    Z = depth[v, u]  # 深度值
    X = (u - cx) * Z / fx  # X座標
    Y = (v - cy) * Z / fy  # Y座標
    coord_text = f"({X:.2f},{Y:.2f},{Z:.2f})"

    # 將左右原始圖像左右排列
    left_color = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    right_color = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    stereo_pair = np.hstack([left_color, right_color])

    # 深度圖轉為可視化圖像（色彩映射）
    depth_vis = np.clip(depth, 0, np.percentile(depth, 99))
    depth_vis = ((depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # 在深度圖上標註 3D 座標與點
    cv2.circle(depth_vis, (u, v), 6, (0, 0, 255), -1)  # 紅色圓點
    cv2.putText(depth_vis, coord_text, (u+10, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # 上下排列 stereo_pair 與 depth_vis
    h, w, _ = stereo_pair.shape
    depth_vis_resized = cv2.resize(depth_vis, (w, h))
    final_img = np.vstack([stereo_pair, depth_vis_resized])
    cv2.imwrite("stereo_demo_result.png", final_img)

    print(f"像素({u},{v}) 的 3D 座標為: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
    print("Demo 完成，已輸出 stereo_demo_result.png、disparity_demo.png 與 depth_demo.npy")
