import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Dataset folders
normal_dir = "data/test/Normal"
tb_dir = "data/test/TB"

# Reproducible random selection
random.seed(42)

# Get image files
normal_files = [
    f for f in os.listdir(normal_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

tb_files = [
    f for f in os.listdir(tb_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Select 4 images from each class
normal_samples = random.sample(normal_files, 4)
tb_samples = random.sample(tb_files, 4)

# Create comparison figure
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Normal images
for i, filename in enumerate(normal_samples):
    image_path = os.path.join(normal_dir, filename)
    image = Image.open(image_path).convert("RGB")

    axes[0, i].imshow(image)
    axes[0, i].set_title(f"Actual: Normal\n{filename}", fontsize=9)
    axes[0, i].axis("off")

# TB images
for i, filename in enumerate(tb_samples):
    image_path = os.path.join(tb_dir, filename)
    image = Image.open(image_path).convert("RGB")

    axes[1, i].imshow(image)
    axes[1, i].set_title(f"Actual: TB\n{filename}", fontsize=9)
    axes[1, i].axis("off")

plt.suptitle(
    "Dataset Verification: Normal vs TB",
    fontsize=16
)

plt.tight_layout()

# Save because WSL does not support plt.show() normally
output_file = "dataset_verification.png"
plt.savefig(output_file, dpi=150)
plt.close()

print("Dataset verification completed!")
print(f"Normal images checked: {len(normal_samples)}")
print(f"TB images checked: {len(tb_samples)}")
print(f"Saved visualization: {output_file}")

print("\nNormal samples:")
for filename in normal_samples:
    print(filename)

print("\nTB samples:")
for filename in tb_samples:
    print(filename)