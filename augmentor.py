# ============================================================
# augmentor.py — Light, CPU-Friendly Chest X-ray Augmentation
# ============================================================

import os
import glob
import random
import time
import shutil
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (48, 48)

# Keep this LOW — augmentation is for balance, not explosion
TARGET_IMAGES_PER_CLASS = 2500   # adjust if needed

# Light, medically safe augmentation
augmenter = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.05,
    brightness_range=(0.9, 1.1)
)


# ============================================================
# BALANCED OFFLINE AUGMENTATION FUNCTION
# ============================================================

def augment_to_target(original_dir, augmented_dir, class_names):

    for cls in class_names:
        print(f"\nProcessing class: {cls}")

        src = os.path.join(original_dir, cls)
        dst = os.path.join(augmented_dir, cls)
        os.makedirs(dst, exist_ok=True)

        originals = [
            f for f in glob.glob(os.path.join(src, "*"))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Copy originals once
        for img in originals:
            dst_img = os.path.join(dst, os.path.basename(img))
            if not os.path.exists(dst_img):
                shutil.copy(img, dst_img)

        current_count = len(os.listdir(dst))
        print(f"Initial images: {current_count}")

        while current_count < TARGET_IMAGES_PER_CLASS:
            img_path = random.choice(originals)

            img = load_img(img_path, target_size=IMG_SIZE)
            arr = img_to_array(img)[None, ...]

            aug_img = next(augmenter.flow(arr, batch_size=1))[0]

            filename = f"aug_{int(time.time()*1e6)}.jpg"
            save_path = os.path.join(dst, filename)

            tf.keras.preprocessing.image.save_img(save_path, aug_img)
            current_count += 1

        print(f"Final images: {current_count}")

    print("\nLight offline augmentation completed successfully")
