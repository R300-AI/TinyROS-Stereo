import numpy as np
import cv2

class Camera(cv2.VideoCapture):
    def __init__(self, index, shape = (480, 640, 3)):
        super().__init__(index)
        self.shape = shape
        print(f"[Capture] 初始化攝影機 index={index}, shape={shape}")
    
    def read(self):
        ret, frame = super().read()
        if not ret:
            print(f"[Capture] 讀取失敗，回傳隨機遮罩 shape={self.shape}")
            return np.random.randint(0, 256, self.shape, dtype=np.uint8)
        print(f"[Capture] 讀取成功，frame shape={frame.shape}")
        return frame

class ChessBaord():
    def __init__(self, size):
        object_points = np.zeros((np.prod(size), 3), np.float32)
        object_points[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
        self.object_points = object_points
        self.size = size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 64, 1e-6)

    def show(self):
        rows, cols = self.size[1], self.size[0]
        board = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                board[i, j] = (i + j) % 2
        plt.imshow(board, cmap='gray', interpolation='nearest')
        plt.title(f'Chessboard {cols}x{rows}')
        plt.axis('off')
        plt.show()
    
    def sampling(self, size=1):
        return [self.object_points for _ in range(size)]