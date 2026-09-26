import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

from b_data_loader import (
    train_dataset,
    validation_dataset
)

# Load ResNet50 with ImageNet pretrained weights
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# Class weights
normal_count = 359
tb_count = 1745
total_count = normal_count + tb_count

class_weights = {
    0: total_count / (2 * normal_count),
    1: total_count / (2 * tb_count)
}

print("\nClass weights:")
print(f"Normal: {class_weights[0]:.2f}")
print(f"TB: {class_weights[1]:.2f}")

# Training
EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights
)

# Save trained model
model.save("resnet50_corrected.keras")

print("\nResNet50 training completed!")
print("Model saved as: resnet50_corrected.keras")