import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
def load_data(img_dir, mask_dir, size=(256,256)):
    print("Loading U-Net training data...")
    images, masks = [], []
    for cls in ['glioma', 'meningioma', 'pituitary']:
        c_img = os.path.join(img_dir, cls)
        c_msk = os.path.join(mask_dir, cls)
        
        if not os.path.exists(c_msk): continue
        
        for fname in os.listdir(c_msk)[:150]: # First 150 per class for speed
            m_path = os.path.join(c_msk, fname)
            i_path = os.path.join(c_img, fname)
            
            img = cv2.imread(i_path)
            msk = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and msk is not None:
                img = cv2.resize(img, size) / 255.0
                msk = cv2.resize(msk, size) / 255.0
                masks.append(np.expand_dims(msk, axis=-1))
                images.append(img)
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    X_train, y_train = load_data(
        os.path.join(BASE_DIR, "archive/Training"),
        os.path.join(BASE_DIR, "archive_masks/Training")
    )
    print(f"Loaded {len(X_train)} samples for U-Net.")
    
    if len(X_train) == 0:
        print("ERROR: No data loaded.")
        exit(1)

    model = unet_model()
    # Train for 5 epochs to ensure it completes today and learns the basics
    print("Training Genuine U-Net API...")
    model.fit(X_train, y_train, batch_size=16, epochs=1, validation_split=0.1)
    
    print("Saving best_model.keras...")
    model.save(os.path.join(BASE_DIR, "best_model.keras"))
    print("U-Net training complete!")
