import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from b_data_loader import test_dataset


# Load the newly trained model
model = tf.keras.models.load_model("resnet50_corrected.keras")

print("Model loaded successfully!")


# Get actual labels
y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_dataset
]).ravel()


# Get TB probabilities
y_probability = model.predict(test_dataset).ravel()


# Convert probabilities to predictions
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


# Extract confusion matrix values
tn, fp, fn, tp = cm.ravel()

print("\nDetailed Results:")
print(f"True Negatives (Normal correctly predicted): {tn}")
print(f"False Positives (Normal predicted as TB): {fp}")
print(f"False Negatives (TB predicted as Normal): {fn}")
print(f"True Positives (TB correctly predicted): {tp}")


# TB sensitivity / recall
tb_sensitivity = tp / (tp + fn)

# Normal specificity
normal_specificity = tn / (tn + fp)

# ROC-AUC
roc_auc = roc_auc_score(y_true, y_probability)


print("\nKey Metrics:")
print(f"TB Sensitivity (Recall): {tb_sensitivity:.4f}")
print(f"Normal Specificity: {normal_specificity:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


print("\nTest evaluation completed!")