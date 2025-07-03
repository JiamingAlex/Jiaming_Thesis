import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像路径（换成你自己的）
s0 = cv2.imread("S0.png", cv2.IMREAD_GRAYSCALE)
s1 = cv2.imread("S1.png", cv2.IMREAD_GRAYSCALE)
s2 = cv2.imread("S2.png", cv2.IMREAD_GRAYSCALE)

# 自动裁剪为相同大小
min_height = min(s0.shape[0], s1.shape[0], s2.shape[0])
min_width = min(s0.shape[1], s1.shape[1], s2.shape[1])
s0 = s0[:min_height, :min_width]
s1 = s1[:min_height, :min_width]
s2 = s2[:min_height, :min_width]

# 转换为 float 进行偏振计算
s0_f = s0.astype(np.float32) + 1e-6
s1_f = s1.astype(np.float32) - 128
s2_f = s2.astype(np.float32) - 128

# 计算 DoLP（线偏振度）
dolp = np.sqrt(s1_f**2 + s2_f**2) / s0_f
dolp = np.clip(dolp, 0, 1)

# CLAHE增强 S0
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
s0_clahe = clahe.apply(s0)

# 将 DoLP 转为掩膜图像
dolp_mask = (dolp * 255).astype(np.uint8)
dolp_mask_eq = clahe.apply(dolp_mask)

# 加权融合
enhanced = cv2.addWeighted(s0_clahe, 0.7, dolp_mask_eq, 0.3, 0)

# 显示结果
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(s0, cmap='gray'); plt.title("Original S0"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(s0_clahe, cmap='gray'); plt.title("CLAHE Only"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(enhanced, cmap='gray'); plt.title("CLAHE + Retinex"); plt.axis("off")
plt.tight_layout(); plt.show()