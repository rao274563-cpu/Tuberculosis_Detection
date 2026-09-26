import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

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

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model("efficientnetb0_corrected.keras")

print("EfficientNetB0 model loaded successfully!")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_dataset
]).ravel()

y_probability = model.predict(test_dataset).ravel()
y_pred = (y_probability >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "TB"],
        zero_division=0
    )
)

tn, fp, fn, tp = cm.ravel()

print("\nDetailed Results:")
print(f"True Negatives (Normal correctly predicted): {tn}")
print(f"False Positives (Normal predicted as TB): {fp}")
print(f"False Negatives (TB predicted as Normal): {fn}")
print(f"True Positives (TB correctly predicted): {tp}")

tb_sensitivity = tp / (tp + fn)
normal_specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_true, y_probability)

print("\nKey Metrics:")
print(f"TB Sensitivity (Recall): {tb_sensitivity:.4f}")
print(f"Normal Specificity: {normal_specificity:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nEfficientNetB0 test evaluation completed!")