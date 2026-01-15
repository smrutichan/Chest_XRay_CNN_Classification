# ============================================================
# kfold_cross_validation.py (No Early Stopping)
# ============================================================

import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"C:\Users\csmru\OneDrive\Desktop\Chest_XRay_CNN_Classification"
DATA_DIR = os.path.join(ROOT, "train")   # TRAIN data only

CLASSES = ["COVID", "NORMAL", "PNEUMONIA"]
IMG_SIZE = (48, 48)
BATCH_SIZE = 128
EPOCHS = 35
N_SPLITS = 5


# ============================================================
# LOAD DATASET INTO MEMORY
# ============================================================

def load_dataset(data_dir, classes, img_size):
    X, y = [], []

    for idx, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            img = load_img(img_path, target_size=img_size)
            img_arr = img_to_array(img) / 255.0

            X.append(img_arr)
            y.append(idx)

    return np.array(X), np.array(y)


X, y = load_dataset(DATA_DIR, CLASSES, IMG_SIZE)
print("Dataset loaded:", X.shape, y.shape)


# ============================================================
# MODEL BUILDER FUNCTION
# ============================================================

def build_model():
    model = Sequential([
        Conv2D(16, (3,3), activation="relu", padding="same",
               input_shape=(48,48,3)),
        MaxPooling2D(2,2),

        Conv2D(32, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(len(CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=["accuracy"]
    )

    return model


# ============================================================
# K-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_accuracies = []
cm_sum = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int32)

fold_no = 1

for train_idx, val_idx in kf.split(X):

    print(f"\n================ FOLD {fold_no} =================")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(CLASSES))
    y_val_cat   = tf.keras.utils.to_categorical(y_val, num_classes=len(CLASSES))

    # Class weights for this fold
    class_weights = {
        0: 1.10,
        1: 1.15,
        2: 1.15
    }

    print("Class weights:", class_weights)

    model = build_model()

    # ===== TRAIN (NO EARLY STOPPING) =====
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=2
    )

    # ===== EVALUATION =====
    loss, acc = model.evaluate(X_val, y_val_cat, verbose=0)
    fold_accuracies.append(acc)

    print(f"Fold {fold_no} Validation Accuracy: {acc:.4f}")

    preds = model.predict(X_val)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(y_val, y_pred)
    cm_sum += cm

    fold_no += 1

# ============================================================
# FINAL K-FOLD RESULTS
# ============================================================

print("\n================ K-FOLD RESULTS ================")
print("Fold Accuracies:", fold_accuracies)
print(f"Mean Accuracy       : {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation  : {np.std(fold_accuracies):.4f}")
print("================================================")


# ============================================================
# AVERAGED CONFUSION MATRIX ACROSS FOLDS
# ============================================================

cm_avg = cm_sum / N_SPLITS

plt.figure(figsize=(7,6))
sns.heatmap(
    cm_avg, annot=True, fmt=".1f", cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)
plt.title("Averaged Confusion Matrix (5-Fold Cross Validation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
