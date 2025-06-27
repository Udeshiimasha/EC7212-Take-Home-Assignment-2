import cv2
import numpy as np
import os

# ---------- 1. Create Synthetic Image with Two Objects and Background ----------
def create_synthetic_image():
    img = np.ones((200, 200), dtype=np.uint8) * 50  # background (pixel value = 50)
    cv2.rectangle(img, (50, 50), (90, 140), 120, -1)  # object 1 (pixel value = 120)
    cv2.circle(img, (140, 100), 25, 200, -1)         # object 2 (pixel value = 200)
    return img

# ---------- 2. Add Gaussian Noise ----------
def add_gaussian_noise(image, mean=0, std=20):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

# ---------- 3. Apply Otsu’s Thresholding ----------
def apply_otsu(image):
    _, otsu_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_result

# ---------- Task 1 Execution ----------
os.makedirs("task1_otsu_output", exist_ok=True)

img = create_synthetic_image()
noisy_img = add_gaussian_noise(img)
otsu_img = apply_otsu(noisy_img)

cv2.imwrite("task1_otsu_output/original.png", img)
cv2.imwrite("task1_otsu_output/noisy.png", noisy_img)
cv2.imwrite("task1_otsu_output/otsu_result.png", otsu_img)

cv2.imshow("Original", img)
cv2.imshow("Noisy", noisy_img)
cv2.imshow("Otsu Result", otsu_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#################################################################

import cv2
import numpy as np
import os

def region_growing(image, seed_point, tolerance=10):
    visited = np.zeros_like(image, dtype=np.uint8)
    h, w = image.shape
    seed_val = image[seed_point]
    stack = [seed_point]
    
    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue

        current_val = image[y, x]
        if abs(int(current_val) - int(seed_val)) <= tolerance:
            visited[y, x] = 255  # mark as part of region

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0:
                        stack.append((ny, nx))

    return visited

# ---------- Task 2 Execution ----------
os.makedirs("task2_region_growing_output", exist_ok=True)

image = create_synthetic_image()
seed1 = (60, 60)   # inside rectangle
seed2 = (140, 100) # inside circle

region1 = region_growing(image, seed1, tolerance=20)
region2 = region_growing(image, seed2, tolerance=20)

cv2.imwrite("task2_region_growing_output/original.png", image)
cv2.imwrite("task2_region_growing_output/region1_from_seed1.png", region1)
cv2.imwrite("task2_region_growing_output/region2_from_seed2.png", region2)

cv2.imshow("Original", image)
cv2.imshow("Region 1", region1)
cv2.imshow("Region 2", region2)
cv2.waitKey(0)
cv2.destroyAllWindows()
