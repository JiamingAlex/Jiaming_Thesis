import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取灰度图像
img = cv2.imread("S0.png", cv2.IMREAD_GRAYSCALE)

# 创建 CLAHE 对象（可调节 clipLimit 和 tileGridSize 参数）
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_img = clahe.apply(img)

# 显示对比效果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original S0")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("CLAHE Enhanced")
plt.imshow(enhanced_img, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
