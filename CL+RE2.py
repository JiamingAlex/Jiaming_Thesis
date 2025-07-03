import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread("examples/45_degree.bmp", cv2.IMREAD_GRAYSCALE)

# 单尺度 Retinex 实现
def single_scale_retinex(img, sigma=30):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log1p(img.astype(np.float32)) - np.log1p(blur.astype(np.float32) + 1)
    return retinex

# Step 1: Retinex
retinex_result = single_scale_retinex(img)
retinex_norm = cv2.normalize(retinex_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Step 2: CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(retinex_norm)

# 显示对比
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(retinex_norm, cmap='gray')
plt.title("CLAHE")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(enhanced, cmap='gray')
plt.title("DoLP + CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()