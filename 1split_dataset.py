import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "Dataset of Tuberculosis Chest X-rays Images"
classes = {
    "Normal Chest X-rays": "Normal",
    "TB Chest X-rays": "TB"
}

for source_folder, class_name in classes.items():
    source_path = os.path.join(dataset_path, source_folder)

    image = [
        file for file in os.listdir(source_path)
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    train, temp = train_test_split(
        image,
        test_size=0.30,
        random_state=42
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42
    )

    splits = {
        "train" : train,
        "validation" : validation,
        "test" : test
    }

    for split_name, split_images in splits.items():
        destination = os.path.join("data", split_name, class_name)
        os.makedirs(destination, exist_ok=True)
        
        for image in split_images:
                source = os.path.join(source_path, image)
                destination_path = os.path.join(destination, image)
        
                if not os.path.exists(destination_path):
                    shutil.copy2(source, destination_path)
        
        print(f"{class_name} - {split_name}: {len(split_images)} images")
print("Dataset splitting completed!")    
