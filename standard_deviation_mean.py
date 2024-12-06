import numpy as np
from PIL import Image
import os
from torchvision.transforms import ToTensor

DATA_ROOT = "/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data"
IMAGE_DIR = f"{DATA_ROOT}/brightfield"

all_images = []
for subdir, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith(".tif"):  # Adjust extension as needed
            img_path = os.path.join(subdir, file)
            image = np.array(Image.open(img_path))
            all_images.append(image)

all_images = np.stack(all_images)  # Shape: (num_images, H, W)
mean = all_images.mean() / 255.0  # Normalize to [0, 1] range
std = all_images.std() / 255.0

print(f"Dataset Mean: {mean}, Dataset Std: {std}")
