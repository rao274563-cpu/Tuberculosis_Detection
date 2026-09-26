import pandas as pd

results = {
    "Model": [
        "ResNet50",
        "VGG16",
        "EfficientNetB0"
    ],
    "TB Sensitivity": [
        0.9813,
        0.9227,
        0.9387
    ],
    "Normal Specificity": [
        1.0000,
        1.0000,
        0.9744
    ],
    "ROC-AUC": [
        0.9998,
        0.9893,
        0.9918
    ],
    "False Negatives": [
        7,
        29,
        23
    ],
    "False Positives": [
        0,
        0,
        2
    ]
}

comparison = pd.DataFrame(results)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

print("\nPrimary Metric: TB Sensitivity")
print("Secondary Metrics: Normal Specificity and ROC-AUC")