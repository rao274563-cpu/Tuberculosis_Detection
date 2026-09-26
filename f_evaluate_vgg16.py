import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from tensorflow.keras.applications.vgg16 import preprocess_input

# Test dataset
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# VGG16 preprocessing
test_dataset = test_dataset.map(
    lambda images, labels: (preprocess_input(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Load trained model
model = tf.keras.models.load_model("vgg16_corrected.keras")

print("VGG16 model loaded successfully!")

# True labels
y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_dataset
]).ravel()

# Predictions
y_probability = model.predict(test_dataset).ravel()
y_pred = (y_probability >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "TB"],
        zero_division=0
    )
)

# Detailed results
tn, fp, fn, tp = cm.ravel()

print("\nDetailed Results:")
print(f"True Negatives (Normal correctly predicted): {tn}")
print(f"False Positives (Normal predicted as TB): {fp}")
print(f"False Negatives (TB predicted as Normal): {fn}")
print(f"True Positives (TB correctly predicted): {tp}")

# Key metrics
tb_sensitivity = tp / (tp + fn)
normal_specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_true, y_probability)

print("\nKey Metrics:")
print(f"TB Sensitivity (Recall): {tb_sensitivity:.4f}")
print(f"Normal Specificity: {normal_specificity:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nVGG16 test evaluation completed!")