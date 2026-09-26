import tensorflow as tf

img_size = (224, 224)
batch_size = 32

# dataset folder path
train_dir = "data/train"
validation_dir = "data/validation"
test_dir = "data/test"

#Loading training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size = img_size,
    batch_size = batch_size,
    label_mode = "binary",
    shuffle = True,
    seed = 42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size = img_size,
    batch_size = batch_size,
    label_mode = "binary",
    shuffle = False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = img_size,
    batch_size = batch_size,
    label_mode = "binary",
    shuffle = False
)

#Display the class names
# print("Class names:", train_dataset.class_names)

# print("Number of training batches:", tf.data.experimental.cardinality(train_dataset).numpy())
# print("Number of validation batches:", tf.data.experimental.cardinality(validation_dataset).numpy())
# print("Number of test batches:", tf.data.experimental.cardinality(test_dataset).numpy())

from tensorflow.keras.applications.resnet50 import preprocess_input

train_dataset = train_dataset.map(
    lambda images, labels: (preprocess_input(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (preprocess_input(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    lambda images, labels: (preprocess_input(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

#Improve data pipeline performance
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

#print("Preprocessing completed successfully!")
print("ResNet50 preprocessing completed successfully!")


#Visually verify one x-ray
import matplotlib.pyplot as plt

#Get one batch from the training dataset
images, labels = next(iter(train_dataset))


#Select the first image and its label
image = images[0]
label = labels[0].numpy().item()


#Define class names
class_names = ["Normal", "TB"]
#Convert binary label into class name
class_name = class_names[int(label)]

#Display the image
plt.figure(figsize=(6,6))
plt.imshow(image)
plt.title(f"Class: {class_name}")   
plt.axis("off")

plt.savefig("sample_image.png")
plt.close()

print(f"Sample X-ray saved as: sample_image.png")
print(f"Sample label: {class_name}")