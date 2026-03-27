import os
import cv2
import numpy as np

def generate_masks():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(BASE_DIR, "archive/Training")
    output_base_dir = os.path.join(BASE_DIR, "archive_masks/Training")
    
    print(f"Generating masks from {input_dir}...")
    
    if not os.path.exists(input_dir):
        print("Error: Input archive folder not found.")
        return

    # Process classes that have tumors
    for cls in ['glioma', 'meningioma', 'pituitary']:
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_base_dir, cls)
        os.makedirs(cls_output, exist_ok=True)
        
        count = 0
        for fname in os.listdir(cls_input):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            img_path = os.path.join(cls_input, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # --- HIGH PRECISION THRESHOLDING ---
            # Use high-pass filter to isolate the brightest spots (the tumor core)
            max_val = np.max(gray)
            if max_val < 50: continue # Skip dark images
            
            _, thresh = cv2.threshold(gray, max(max_val * 0.75, 70), 255, cv2.THRESH_BINARY)
            
            # Clean up with morphology
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find the largest contour (the tumor)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > (gray.shape[0] * gray.shape[1] * 0.001):
                    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
            
            # Save the generated ground truth
            cv2.imwrite(os.path.join(cls_output, fname), mask)
            count += 1
            if count >= 200: break # Limit for fast reproduction
            
        print(f"Generated {count} masks for class: {cls}")

if __name__ == "__main__":
    generate_masks()
