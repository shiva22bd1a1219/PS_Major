import os
import cv2
import numpy as np
from glob import glob

def generate_mask(image_path, mask_path):
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    
    # 1. Thresholding to find highest intensity regions (typically tumor)
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological operations to clean up noise
    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(img)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Avoid masks that are empty or literally the entire image (skull)
        if cv2.contourArea(largest_contour) < 0.9 * img.shape[0] * img.shape[1]:
            cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    
    cv2.imwrite(mask_path, mask)
    return True

base_dir = "C:/Users/Shiva/Desktop/mlml/PS!major/intelliscan/intelliscan backend/archive/Training"
out_dir = "C:/Users/Shiva/Desktop/mlml/PS!major/intelliscan/intelliscan backend/archive_masks/Training"

print("Starting mask generation...")
count = 0
for class_name in ['glioma', 'meningioma', 'pituitary']:
    img_folder = os.path.join(base_dir, class_name)
    out_folder = os.path.join(out_dir, class_name)
    os.makedirs(out_folder, exist_ok=True)
    
    images = glob(os.path.join(img_folder, "*.*"))
    # Generate masks for first 200 images per class to keep it lightning fast for today
    for img_path in images[:200]:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(out_folder, filename)
        if generate_mask(img_path, mask_path):
            count += 1

print(f"Generated {count} masks for U-Net training.")
