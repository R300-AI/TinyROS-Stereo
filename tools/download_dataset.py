import os
from datasets import load_dataset
from tqdm import tqdm

# 自訂儲存路徑（例如："./holopix50k_images"）
save_dir = "./data/holopix50k_images"
os.makedirs(save_dir, exist_ok=True)

# 載入所有 split（train, validation, test）
splits = ["train", "validation", "test"]
dataset = load_dataset("ernestchu/holopix50k")

# 儲存影像
for split in splits:
    split_dir = os.path.join(save_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"正在儲存 {split} split，共 {len(dataset[split])} 筆資料...")
    for i, sample in enumerate(tqdm(dataset[split])):
        left = sample["left_image"]
        right = sample["right_image"]
        left.save(os.path.join(split_dir, f"{i:05d}_left.jpg"))
        right.save(os.path.join(split_dir, f"{i:05d}_right.jpg"))
