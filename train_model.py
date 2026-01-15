# ============================================================
# train_model.py — Chest X-ray CNN Training Pipeline (FINAL)
# ============================================================

import os
import glob
import random
import time
import shutil
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array


from augmentor import augment_to_target


# ============================================================
# 1. PATH DEFINITIONS
# ============================================================

ROOT = r"C:\Users\csmru\OneDrive\Desktop\Chest_XRay_CNN_Classification"

ORIGINAL_DIR  = os.path.join(ROOT, "original_dataset")
AUGMENTED_DIR = os.path.join(ROOT, "augmented_dataset")

TRAIN = os.path.join(ROOT, "train")
VALID = os.path.join(ROOT, "valid")
TEST  = os.path.join(ROOT, "test")

CLASSES  = ["COVID", "NORMAL", "PNEUMONIA"]
IMG_SIZE = (48, 48)
BATCH_SIZE = 128


# ============================================================
# 2. ENSURE REQUIRED FOLDER STRUCTURE EXISTS
# ============================================================

for folder in [AUGMENTED_DIR, TRAIN, VALID, TEST]:
    os.makedirs(folder, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)


# ============================================================
# 3. ONE-TIME OFFLINE AUGMENTATION
# ============================================================

def is_augmented_dataset_empty(aug_dir, class_names):
    for cls in class_names:
        cls_dir = os.path.join(aug_dir, cls)
        if os.path.exists(cls_dir) and len(os.listdir(cls_dir)) > 0:
            return False
    return True


if is_augmented_dataset_empty(AUGMENTED_DIR, CLASSES):
    print("Running one-time augmentation...")
    augment_to_target(ORIGINAL_DIR, AUGMENTED_DIR, CLASSES)
else:
    print("Augmented dataset already exists. Skipping augmentation.")


# ============================================================
# 4. SAFE COPY FUNCTION
# ============================================================

def safe_copy(src, dst_folder):
    dst = os.path.join(dst_folder, os.path.basename(src))
    if not os.path.exists(dst):
        shutil.copy(src, dst)


# ============================================================
# 5. DATA SPLIT (70 / 20 / 10)
# ============================================================

random.seed(42)

for cls in CLASSES:
    images = glob.glob(os.path.join(AUGMENTED_DIR, cls, "*"))
    random.shuffle(images)

    n = len(images)
    n_train = int(0.7 * n)
    n_valid = int(0.2 * n)

    for img in images[:n_train]:
        safe_copy(img, os.path.join(TRAIN, cls))
    for img in images[n_train:n_train + n_valid]:
        safe_copy(img, os.path.join(VALID, cls))
    for img in images[n_train + n_valid:]:
        safe_copy(img, os.path.join(TEST, cls))

print("Dataset split complete (70 / 20 / 10)")


# ============================================================
# 6. IMAGE DATA GENERATORS
# ============================================================

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TRAIN,
    target_size=IMG_SIZE,
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VALID,
    target_size=IMG_SIZE,
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST,
    target_size=IMG_SIZE,
    classes=CLASSES,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="categorical"
)


# ============================================================
# 7. CLASS WEIGHTS (CONFUSION MATRIX BALANCING)
# ============================================================

class_weights = {
    0: 1.15,  # COVID 
    1: 1.1,   # NORMAL 
    2: 1.1    # PNEUMONIA 
}

print("Using class weights:", class_weights)


# ============================================================
# 8. CNN MODEL (FINAL)
# ============================================================

model = Sequential([
    Conv2D(16, (3,3), activation="relu", padding="same",
           input_shape=(48,48,3)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation="relu", padding="same"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),   # strengthened head
    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer=Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 9. TRAINING WITH CUSTOM EARLY STOPPING
# ============================================================

class SmartEarlyStop(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, min_epoch=20):
        super().__init__()
        self.patience = patience
        self.min_epoch = min_epoch
        self.same = 0
        self.last = None

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        val_acc = logs.get("val_accuracy")

        if val_acc == self.last:
            self.same += 1
        else:
            self.same = 0

        self.last = val_acc

        if epoch >= self.min_epoch and self.same >= self.patience:
            print(
                f"Training stopped: validation accuracy unchanged for "
                f"{self.patience} consecutive epochs."
            )
            self.model.stop_training = True


callback = SmartEarlyStop(patience=5, min_epoch=20)

start_time = time.time()
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=50,
    callbacks=[callback],
    verbose=2
)
training_time = time.time() - start_time

# ============================================================
# 10. CONFUSION MATRIX EVALUATION (TEST SET ONLY)
# ============================================================
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
preds = model.predict(test_gen)
y_pred = preds.argmax(axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)
plt.title("Confusion Matrix - Chest X-ray CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# ============================================================
# 11. TRAINING CURVES (LOSS vs ACCURACY)
# ============================================================

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 12. TRAINING SUMMARY
# ============================================================

print("\n================ TRAINING SUMMARY ================")
print(f"Total Training Time      : {training_time:.2f} seconds")
print(f"Final Training Accuracy  : {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy            : {test_acc:.4f}")
print("==================================================")

# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model.save(os.path.join(ROOT, "chest_xray_cnn_model.keras"))
print("Model saved successfully.")

# ============================================================
# 14. SINGLE IMAGE PREDICTION
# ============================================================


IMG_PATH = r"C:\Users\csmru\OneDrive\Desktop\Chest_XRay_CNN_Classification\PNEUMONIA_27.png"

img = load_img(IMG_PATH, target_size=IMG_SIZE)
img_arr = img_to_array(img) / 255.0
img_arr = np.expand_dims(img_arr, axis=0)

pred = model.predict(img_arr)[0]
pred_class = CLASSES[np.argmax(pred)]
confidence = np.max(pred) * 100

plt.imshow(img)
plt.axis("off")
plt.title(f"Prediction: {pred_class} ({confidence:.2f}%)")
plt.show()

print("Prediction probabilities:")
for cls, p in zip(CLASSES, pred):
    print(f"{cls}: {p*100:.2f}%")

