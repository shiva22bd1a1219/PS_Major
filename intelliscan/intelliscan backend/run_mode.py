import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
import torch
import tensorflow as tf
import io
from PIL import Image
from patchify import patchify
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision import models
import base64

# ===== FIX MASK PATH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_PATH = os.path.join(BASE_DIR, "output_mask.png")

MASK_FOLDER = os.path.join(BASE_DIR, "static_mask")
os.makedirs(MASK_FOLDER, exist_ok=True)
# ---------------- CONFIG ----------------
cf = {
    "image_size": 256,
    "num_channels": 3,
    "patch_size": 16,
}

cf["num_patches"] = (cf["image_size"]**2)//(cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"]*cf["patch_size"]*cf["num_channels"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smooth = 1e-15

# ---------------- Dice ----------------
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.*intersection + smooth) / (
        tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)+smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# ---------------- LOAD MODELS ----------------
# Hardcoded class names to avoid loading 151MB of training data
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("Loading ResNet...")
vit_model = models.resnet18()
vit_model.fc = nn.Linear(vit_model.fc.in_features, 4)
RESNET_PATH = os.path.join(BASE_DIR, "resnet_model.pth")

try:
    checkpoint = torch.load(RESNET_PATH, map_location=device)
    vit_model.load_state_dict(checkpoint)
    print("ResNet weights loaded!")
except Exception as e:
    print("Could not load ResNet weights yet: ", e)

vit_model.to(device)
vit_model.eval()

# ----- UNET LOADING (HYBRID SEGMENTATION) -----
# Check for both possible names from your training sessions
UNET_MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
if not os.path.exists(UNET_MODEL_PATH):
    UNET_MODEL_PATH = os.path.join(BASE_DIR, "best_model(1).keras")

unet_model = None
if os.path.exists(UNET_MODEL_PATH):
    try:
        print(f"Restoring UNet model from {os.path.basename(UNET_MODEL_PATH)}...")
        unet_model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
        print("UNet successfully restored and working!")
    except Exception as e:
        print(f"Error loading UNet model weights: {e}")
else:
    print("UNet model not found. Using high-precision Unsupervised OpenCV Segmenter.")

# Removed ImageFolder dependency

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- FUNCTIONS ----------------
# def preprocess_patchify(image):
#     image = cv2.resize(image,(cf["image_size"],cf["image_size"]))
#     image_norm = image/255.0

#     patch_shape=(cf["patch_size"],cf["patch_size"],cf["num_channels"])
#     patches=patchify(image_norm,patch_shape,cf["patch_size"])
#     patches=np.reshape(patches,cf["flat_patches_shape"])
#     patches=patches.astype(np.float32)

#     return np.expand_dims(patches,axis=0), image_norm

def predict_segmentation(image_cv2):
    height, width = image_cv2.shape[:2]
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    
    # CASE 1: USE UNET (If restored from Recycle Bin)
    if unet_model:
        try:
            # Preprocess for UNet (256x256)
            img_input = cv2.resize(image_cv2, (256, 256)) / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            
            # Predict
            pred_unet = unet_model.predict(img_input, verbose=0)[0]
            pred_unet = (pred_unet > 0.5).astype(np.uint8)
            
            # SANITY CHECK: UNet must detect a significant mass that isn't just at the boundary
            if np.sum(pred_unet) > 20: # Must have some content
                # Check for "Edge Hugging" - if more than 60% of mask is near edges, it's likely skull
                edge_pad = 20
                edge_mask = np.ones((256, 256), dtype=np.uint8)
                edge_mask[edge_pad:-edge_pad, edge_pad:-edge_pad] = 0
                if np.sum(pred_unet * edge_mask) / np.sum(pred_unet) < 0.6:
                    mask = cv2.resize(pred_unet, (width, height))
                    return (mask > 0).astype(np.float32), (image_cv2 / 255.0).astype(np.float32)
                else:
                    print("UNet predicted edge-heavy mask (likely skull). Falling back to OpenCV.")
        except Exception as e:
            print(f"UNet Inference Error: {e}. Falling back to OpenCV logic.")

    # CASE 2: ADVANCED OPENCV SEGMENTER (Refined for Skull Stripping)
    blur_kernel = (width // 15) | 1
    blur_kernel = max(blur_kernel, 21)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    max_val = np.max(blurred)
    if max_val < 50:
        return np.zeros_like(gray, dtype=np.float32), (image_cv2 / 255.0).astype(np.float32)

    # Use a relative threshold but biased towards the absolute brightest spots
    thresh_val = max(max_val * 0.70, 70)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    img_center = (width // 2, height // 2)
    
    if contours:
        scored_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < (height * width * 0.001): continue # Too small
            
            # 1. Solidity Filter (Tumors are blobs, skull is thin)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # 2. Center Bias (Tumors are usually not touching the image border)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            dist_from_center = np.sqrt((cX - img_center[0])**2 + (cY - img_center[1])**2)
            max_dist = np.sqrt(img_center[0]**2 + img_center[1]**2)
            center_score = 1.0 - (dist_from_center / max_dist) # 1.0 at center, 0.0 at corner
            
            # 3. Peak Intensity (Prioritize the absolute brightest region)
            temp_mask = np.zeros_like(gray)
            cv2.drawContours(temp_mask, [c], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=temp_mask)[0]
            
            # Final Score
            total_score = (solidity * 0.4) + (center_score * 0.4) + (mean_val / 255.0 * 0.2)
            scored_contours.append((total_score, c))
            
        if scored_contours:
            # Pick the highest scoring "Tumor-like" object
            best_c = max(scored_contours, key=lambda x: x[0])[1]
            cv2.drawContours(mask, [best_c], -1, 255, thickness=cv2.FILLED)
            
    # Normalize and return
    return (mask > 0).astype(np.float32), (image_cv2 / 255.0).astype(np.float32)

def overlay_mask(image,mask):
    image_uint8=(image*255).astype(np.uint8)
    mask_uint8=(mask*255).astype(np.uint8)

    mask_col=np.zeros_like(image_uint8)
    mask_col[:,:,1]=mask_uint8

    return cv2.addWeighted(image_uint8,1.0,mask_col,0.5,0)

def predict_class(image_pil):
    tensor=data_transforms(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out=vit_model(tensor)
        probs=torch.nn.functional.softmax(out[0],dim=0)

    conf,pred=torch.max(probs,0)
    return pred.item(),conf.item(),probs.cpu().numpy()

import uuid

# ---------------- FLASK ----------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():

    file=request.files["image"]
    bytes_data=file.read()
    np_bytes=np.frombuffer(bytes_data,np.uint8)

    image_cv2 = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    if image_cv2 is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        image_pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400
    # image_cv2=cv2.imdecode(np_bytes,cv2.IMREAD_COLOR)
    # image_pil=Image.open(io.BytesIO(bytes_data)).convert("RGB")

    pred_mask,norm_image=predict_segmentation(image_cv2)
    overlay=overlay_mask(norm_image,pred_mask)

    pred_class,conf,probs=predict_class(image_pil)
    
    # Save unique mask to fix caching bug
    unique_id = str(uuid.uuid4())[:8]
    mask_filename = f"mask_{unique_id}.png"
    mask_filepath = os.path.join(MASK_FOLDER, mask_filename)
    cv2.imwrite(mask_filepath, overlay)
    
    # Calculate Area using ratio of mask to real resolution
    height, width = image_cv2.shape[:2]
    # Rough approximation: assuming the MRI image represents a 25x25 cm brain scan area
    pixels_per_cm2 = (height * width) / (25 * 25)
    total_tumor_pixels = np.sum(pred_mask)
    tumor_area_cm2 = round(total_tumor_pixels / pixels_per_cm2, 2)
    if tumor_area_cm2 == 0:
        tumor_area_cm2 = 0.0
    
    # Determine Risk dynamically based on BOTH classification and size
    predicted_label = class_names[pred_class]
    if predicted_label.lower() == 'glioma':
        risk_level = "High Risk (Critical)"
    elif predicted_label.lower() in ['meningioma', 'pituitary']:
        # Even benign tumors become high risk if they grow too large and press on the brain
        if tumor_area_cm2 > 10.0:
            risk_level = "High Risk (Large Mass Compress)"
        else:
            risk_level = "Moderate Risk"
    else:
        risk_level = "Normal (No Tumor)"

    # Encode to base64 for immediate flawless display
    _, buffer = cv2.imencode('.png', overlay)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    mask_data_url = f"data:image/png;base64,{mask_base64}"

    return jsonify({
        "classification":{
            "label":predicted_label,
            "confidence":round(conf,4),
            "probabilities":{
                class_names[i]:round(float(p),4)
                for i,p in enumerate(probs)
            }
        },
        "tumor_area_cm2": tumor_area_cm2,
        "risk_level": risk_level,
        "segmentation_mask_url": mask_data_url
    })

@app.route("/static_mask/<filename>")
def serve_static_mask(filename):
    filepath = os.path.join(MASK_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="image/png")
    return jsonify({"error": "mask not found"}), 404

if __name__ == "__main__":
    app.run(debug=True,port=5003)
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import cv2
# import torch
# import tensorflow as tf
# import os
# import io
# from PIL import Image
# from patchify import patchify
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from transformer import TumorClassifierViT

# import keras
# from keras import layers

# # ------------------ CONFIG ------------------
# cf = {
#     "image_size": 256,
#     "num_channels": 3,
#     "patch_size": 16,
# }
# cf["num_patches"] = (cf["image_size"]**2) // (cf["patch_size"]**2)
# cf["flat_patches_shape"] = (
#     cf["num_patches"],
#     cf["patch_size"] * cf["patch_size"] * cf["num_channels"]
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# smooth = 1e-15

# def dice_coef(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)

# # ------------------ INITIALIZE MODELS ------------------
# UNET_MODEL_PATH = "best_model(1).keras"
# VIT_MODEL_PATH = "best_model.pth"
# CLASS_FOLDER = "archive/Training"
# # UNET_MODEL_PATH = "D:\\intelliscan\\intelliscan backend\\best_model(1).keras"
# # VIT_MODEL_PATH = "D:\\intelliscan\\intelliscan backend\\best_model.pth"
# # CLASS_FOLDER = "D:\\intelliscan\\intelliscan backend\\archive\\Training"

# # import keras

# # unet_model = keras.models.load_model(
# #     UNET_MODEL_PATH,
# #     compile=False,
# #     custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef}
# # )
# unet_model = None
# print("UNet model disabled (invalid file)")
# # unet_model = tf.keras.models.load_model(
# #     UNET_MODEL_PATH, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef}
# # )

# vit_model = TumorClassifierViT(num_classes=4)
# vit_model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
# vit_model.to(device)
# vit_model.eval()

# train_dataset = ImageFolder(CLASS_FOLDER, transform=transforms.ToTensor())
# class_names = train_dataset.classes

# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # ------------------ IMAGE HANDLING ------------------
# def preprocess_patchify(image):
#     image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
#     image_norm = image / 255.0
#     patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
#     patches = patchify(image_norm, patch_shape, cf["patch_size"])
#     patches = np.reshape(patches, cf["flat_patches_shape"])
#     patches = patches.astype(np.float32)
#     return np.expand_dims(patches, axis=0), image_norm

# def predict_segmentation(image):
#     return None, image
# # def predict_segmentation(image):
# #     input_patches, resized_image = preprocess_patchify(image)
# #     pred_mask = unet_model.predict(input_patches)[0]
# #     pred_mask = cv2.resize(pred_mask, (cf["image_size"], cf["image_size"]))
# #     pred_mask = np.where(pred_mask > 0.5, 1.0, 0.0)
# #     return pred_mask, resized_image

# def overlay_mask(image, mask):
#     image_uint8 = (image * 255).astype(np.uint8)
#     mask_uint8 = (mask * 255).astype(np.uint8)
#     mask_colored = np.zeros_like(image_uint8)
#     mask_colored[:, :, 1] = mask_uint8  # Green channel
#     return cv2.addWeighted(image_uint8, 1.0, mask_colored, 0.5, 0)

# def predict_class(image_pil):
#     image_tensor = data_transforms(image_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = vit_model(image_tensor)
#         probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
#     confidence, predicted = torch.max(probabilities, 0)
#     return predicted.item(), confidence.item(), probabilities.cpu().numpy()

# # ------------------ FLASK API ------------------
# app = Flask(__name__)
# CORS(app)

# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     file = request.files['image']

#     # Read file content once
#     file_bytes = file.read()
#     np_bytes = np.frombuffer(file_bytes, np.uint8)

#     # Decode for OpenCV
#     image_cv2 = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
#     if image_cv2 is None:
#         return jsonify({'error': 'Invalid image format (cv2)'}), 400

#     # Decode for PIL
#     try:
#         image_pil = Image.open(io.BytesIO(file_bytes)).convert('RGB')
#     except Exception as e:
#         return jsonify({'error': f'Invalid image format (PIL): {str(e)}'}), 400

#     # Run segmentation and classification
#     # pred_mask, norm_image = predict_segmentation(image_cv2)
#     # overlayed = overlay_mask(norm_image, pred_mask)
#     # predicted_class, confidence, prob_array = predict_class(image_pil)

#     # # Save the mask image
#     # overlay_path = "output_mask.png"
#     # cv2.imwrite(overlay_path, overlayed)
#     pred_mask, norm_image = predict_segmentation(image_cv2)

#     return jsonify({
#         "classification": {
#             "label": class_names[predicted_class],
#             "confidence": round(confidence, 4),
#             "probabilities": {class_names[i]: round(float(prob), 4) for i, prob in enumerate(prob_array)}
#         },
#         "segmentation_mask_url": "/mask"
#     })

# @app.route("/mask", methods=["GET"])
# def get_mask():
#     return send_file("output_mask.png", mimetype='image/png')

# if __name__ == "__main__":
#     app.run(debug=True, port=5003)
