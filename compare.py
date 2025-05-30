import cv2
import os

#
image_files = [
    "examples/0_degree.bmp",
    "examples/30_degree.bmp",
    "examples/45_degree.bmp",
    "examples/75_degree.bmp",
]

def compute_laplacian_variance(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"can_not read：{image_path}")
        return None
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var

results = []
for img_file in image_files:
    sharpness = compute_laplacian_variance(img_file)
    if sharpness is not None:
        results.append((img_file, sharpness))

# show clarity rank
results.sort(key=lambda x: x[1], reverse=True)

# print result
print("\nClarity sort:（Laplacian ）:")
for name, score in results:
    print(f"{name}: {score:.2f}")