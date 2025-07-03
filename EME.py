import cv2
import numpy as np

def compute_EME(img, block_size=8):
    """
    计算图像的 EME（局部对比度增强指标）
    参数：
        img: 输入灰度图像 (numpy.ndarray)
        block_size: 分块大小，通常为8
    返回：
        eme: EME值（越高表示局部对比度越强）
    """
    img = img.astype(np.float32)
    h, w = img.shape
    eme = 0
    eps = 1e-5  # 防止除以零

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.size == 0:
                continue
            Imax = block.max()
            Imin = block.min()
            if Imin > 0:
                eme += np.log((Imax + eps) / (Imin + eps))

    eme_value = (20 * eme) / ((h // block_size) * (w // block_size))
    return eme_value

# 示例使用
img1 = cv2.imread("examples/Do.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("examples/Re.png", cv2.IMREAD_GRAYSCALE)
eme_score1 = compute_EME(img1)
eme_score2 = compute_EME(img2)
print("EME =", eme_score1)
print("EME =", eme_score2)