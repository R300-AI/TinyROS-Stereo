import numpy as np
import matplotlib.pyplot as plt
import cv2, os, imageio

def find_chessboard_corners(image_files, chessboard_size, criteria):
    imgpoints = []
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
        else:
            imgpoints.append(None)
    return imgpoints, gray.shape[::-1]

def get_images(camera, display=False, ncols=6, title=None):
    left_dir = os.path.join('data', 'calibrate', camera, 'Left')
    right_dir = os.path.join('data', 'calibrate', camera, 'Right')
    left_images = sorted([os.path.join(left_dir, f) for f in os.listdir(left_dir) if f.lower().endswith('.jpg')])
    right_images = sorted([os.path.join(right_dir, f) for f in os.listdir(right_dir) if f.lower().endswith('.jpg')])
    if display:
        all_images = left_images + right_images
        n_images = len(all_images)
        nrows = int(np.ceil(n_images / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12)) 
        axes = axes.flatten() if n_images > 1 else [axes]
        for ax, img_path in zip(axes, all_images):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(os.path.basename(img_path), fontsize=8)
            ax.axis('off')
        for ax in axes[n_images:]:
            ax.axis('off')
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    return left_images, right_images

def generate_chessboard(size, sample=0, display=False):
    rows, cols = size[1], size[0]
    board = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            board[i, j] = (i + j) % 2
    if display:
        plt.imshow(board, cmap='gray', interpolation='nearest')
        plt.title(f'Chessboard {cols}x{rows}')
        plt.axis('off')
        plt.show()
    
    objp = np.zeros((np.prod(size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

    return [objp for _ in range(sample)]


def plot_pts3d_gif(pts3d, gif_path="./logs/pts3d_rotate.gif", n_frames=36):
    imgs = []
    tmp_dir = "_tmp_gif"
    os.makedirs(tmp_dir, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = pts3d[:,0], pts3d[:,1], pts3d[:,2]
    for i, (x, y, z) in enumerate(pts3d):
        ax.text(x, y, z, f'{i+1}', color='blue')
    ax.scatter(xs, ys, zs, c='r', marker='o')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("3D Points")
    for i in range(n_frames):
        ax.view_init(elev=20, azim=i * 360 / n_frames)
        fname = os.path.join(tmp_dir, f"frame_{i:03d}.png")
        plt.savefig(fname)
        imgs.append(imageio.imread(fname))
    imageio.mimsave(gif_path, imgs, duration=0.05)
    plt.close(fig)
    # 清理暫存圖片
    for fname in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, fname))
    os.rmdir(tmp_dir)
    print(f"GIF已儲存：{gif_path}")



def imshow(left_path, right_path):
    left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
    h = max(left_img.shape[0], right_img.shape[0])
    if left_img.shape[0] != h:
        left_img = cv2.copyMakeBorder(left_img, 0, h - left_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    if right_img.shape[0] != h:
        right_img = cv2.copyMakeBorder(right_img, 0, h - right_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)

    plt.imshow(np.hstack((left_img, right_img)))
    plt.title('Left | Right')
    plt.axis('off')
    plt.show()

def plot_corresponding_points(left_img_path, right_img_path, pts_left, pts_right, marker_color='r'):
    """
    在左右影像上分別標出對應點
    pts_left, pts_right: list of (x, y)
    """
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(left_img_rgb)
    axs[0].set_title('Left Image')
    for i, (x, y) in enumerate(pts_left):
        axs[0].plot(x, y, marker='o', color=marker_color)
        axs[0].text(x+5, y, str(i+1), color=marker_color, fontsize=12)
    axs[0].axis('off')

    axs[1].imshow(right_img_rgb)
    axs[1].set_title('Right Image')
    for i, (x, y) in enumerate(pts_right):
        axs[1].plot(x, y, marker='o', color=marker_color)
        axs[1].text(x+5, y, str(i+1), color=marker_color, fontsize=12)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()